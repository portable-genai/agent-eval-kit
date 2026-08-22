"""The OPT-IN judge: an OpenAI-compatible chat-completions server grades the narrative.

Selected only when a caller names it (see :class:`~agent_eval_kit.judge.JudgeSelection`), never
by default and never as a fallback. The point of the reduced profile is that a laptop can serve
both the subject model and the judge with no cloud and no credentials, so the profile running
the weaker model is no longer the profile with no quality measurement. The point of it being
opt-in is that a gate must still run with no server at all.

**The HTTP client is imported inside the methods that call it, not at module scope.** This
package has already paid for that lesson once: an eager ``from . import gate_client`` in the
package ``__init__`` put ``httpx`` into the import graph of every consumer's decision core,
contradicting the docstring six lines above it. Here the module itself imports nothing but the
standard library, so even ``import agent_eval_kit.local_model_judge`` costs a consumer nothing
until a request is actually made. ``tests/test_core_purity.py`` holds both halves of that.

Fail-closed, everywhere. An unreachable server, a non-2xx reply, a body that is not JSON, a
missing criterion, a duplicate criterion, an extra criterion or a score outside [0, 1] all
raise :class:`~agent_eval_kit.judge.JudgeUnavailableError`. None of them produce a number,
because a score invented by a failure is indistinguishable from a measurement and would be
read as one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from math import isfinite
from typing import Any

from ._urls import require_secure_url
from .judge import CriterionScore, JudgeRequest, JudgeUnavailableError, JudgeVerdict

#: OpenAI-compatible path with NO ``/v1`` prefix, which is what the local MLX server speaks.
#: A server using the conventional prefix is reached by passing ``path="/v1/chat/completions"``.
DEFAULT_CHAT_PATH = "/chat/completions"
DEFAULT_MODELS_PATH = "/models"
DEFAULT_TIMEOUT_SECONDS = 180.0

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z]*\n?|```")

_SYSTEM_PROMPT = (
    "You grade the quality of a written narrative against explicit criteria. "
    "For each criterion, return a score from 0.0 to 1.0: 1.0 when the narrative fully "
    "satisfies it, 0.0 when it does not satisfy it at all. Judge only what the narrative "
    "says. Reply with a single JSON object and nothing else, in the form "
    '{"scores": [{"criterion": "<name>", "score": <number>, "detail": "<short reason>"}]}, '
    "with exactly one entry per criterion you were given."
)


def _criteria_brief(request: JudgeRequest) -> str:
    lines: list[str] = []
    for criterion in request.criteria:
        lines.append(f"- criterion {criterion.name!r}:")
        if criterion.must_cover:
            lines.append(f"    must convey these points: {'; '.join(criterion.must_cover)}")
        if criterion.must_cite:
            lines.append(f"    must cite these references: {'; '.join(criterion.must_cite)}")
        if criterion.must_not_say:
            lines.append(f"    must not claim any of: {'; '.join(criterion.must_not_say)}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the single JSON object out of a chat reply, tolerating fences and thinking blocks."""
    stripped = _FENCE.sub("", _THINK_BLOCK.sub("", text)).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        raise JudgeUnavailableError(f"the judge model replied with no JSON object: {text[:300]!r}")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except ValueError as exc:
        raise JudgeUnavailableError(
            f"the judge model replied with unparseable JSON: {text[:300]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise JudgeUnavailableError("the judge model's JSON reply is not an object")
    return parsed


class LocalModelJudge:
    """Grade narratives with a locally served, OpenAI-compatible chat model.

    ``base_url`` must be https, or plain http on loopback: the candidate narrative and the
    criteria both travel on this connection.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        path: str = DEFAULT_CHAT_PATH,
        models_path: str = DEFAULT_MODELS_PATH,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        auth_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = require_secure_url(base_url, service=type(self).__name__, kind="judge")
        if not model.strip():
            raise ValueError("LocalModelJudge: the judge model id is part of the evidence")
        self._model = model
        self._path = path if path.startswith("/") else f"/{path}"
        self._models_path = models_path if models_path.startswith("/") else f"/{models_path}"
        self._timeout = timeout
        self._auth_headers = dict(auth_headers or {})

    @property
    def name(self) -> str:
        """How this judge attributes its verdicts."""
        return f"local-model:{self._model}"

    def probe(self) -> bool:
        """Whether the server answers at all. Used to SKIP work, never to substitute a score."""
        import httpx

        try:
            response = httpx.get(
                f"{self._base_url}{self._models_path}",
                timeout=min(self._timeout, 5.0),
                headers=self._auth_headers,
            )
        except httpx.HTTPError:
            return False
        return response.status_code // 100 == 2

    def grade(self, request: JudgeRequest) -> JudgeVerdict:
        """Ask the model to grade ``request``, or raise :class:`JudgeUnavailableError`."""
        import httpx

        url = f"{self._base_url}{self._path}"
        try:
            response = httpx.post(
                url,
                json=self._body(request),
                timeout=self._timeout,
                headers=self._auth_headers,
            )
        except httpx.HTTPError as exc:
            raise JudgeUnavailableError(f"judge request to {url} failed: {exc}") from exc
        if response.status_code // 100 != 2:
            raise JudgeUnavailableError(
                f"judge server {url} returned {response.status_code}: {response.text[:300]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise JudgeUnavailableError("judge server returned a non-JSON response") from exc
        return self._parse(request, body)

    def _body(self, request: JudgeRequest) -> dict[str, object]:
        reference = (
            f"Reference material the narrative should rest on:\n{request.reference}\n\n"
            if request.reference
            else ""
        )
        user = (
            f"{reference}Narrative to grade:\n{request.candidate}\n\n"
            f"Criteria:\n{_criteria_brief(request)}"
        )
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            # Deterministic decoding as far as the server allows it: a judge that answers
            # differently on a rerun cannot be a promotion bar.
            "temperature": 0.0,
            "stream": False,
        }

    def _parse(self, request: JudgeRequest, body: object) -> JudgeVerdict:
        if not isinstance(body, dict):
            raise JudgeUnavailableError("judge server response must be a JSON object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise JudgeUnavailableError("judge server response carries no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise JudgeUnavailableError("judge server response carries no message content")

        payload = _extract_json_object(content)
        rows = payload.get("scores")
        if not isinstance(rows, list) or not rows:
            raise JudgeUnavailableError(
                "the judge model returned no scores; an empty verdict is not a pass"
            )
        expected = {criterion.name: criterion.weight for criterion in request.criteria}
        scores: list[CriterionScore] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise JudgeUnavailableError(f"judge score {index} is not an object")
            name = row.get("criterion")
            if not isinstance(name, str) or name not in expected:
                raise JudgeUnavailableError(
                    f"judge score {index} names {name!r}, which is not one of the criteria "
                    f"it was given ({', '.join(sorted(expected))})"
                )
            if name in seen:
                raise JudgeUnavailableError(f"the judge graded {name!r} more than once")
            raw = row.get("score")
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise JudgeUnavailableError(f"judge score for {name!r} is not a number")
            value = float(raw)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise JudgeUnavailableError(f"judge score for {name!r} is {raw!r}, outside [0, 1]")
            detail = row.get("detail", "")
            seen.add(name)
            scores.append(
                CriterionScore(
                    criterion=name,
                    score=value,
                    weight=expected[name],
                    detail=str(detail)[:300],
                )
            )
        missing = sorted(set(expected) - seen)
        if missing:
            raise JudgeUnavailableError(
                f"the judge did not grade {', '.join(missing)}; a partial verdict would be "
                "scored against a floor that assumes the whole criteria set"
            )
        return JudgeVerdict(graded_by=self.name, scores=tuple(scores))


__all__ = [
    "DEFAULT_CHAT_PATH",
    "DEFAULT_MODELS_PATH",
    "DEFAULT_TIMEOUT_SECONDS",
    "LocalModelJudge",
]
