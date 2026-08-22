"""A promotion-gate HTTP client: one contract for a service's ``--mode gate``.

At promotion, a model or agent's quality is checked against a shared promotion-gate service (an
"AI quality" / model-risk service). Two calls, both fail-closed:

* ``evaluate`` -> ``POST /v1/evaluations {target, dataset_id, bundle}`` -> :class:`EvalReport`,
  parsed from the response ``results`` list (NOT ``metrics``), each row carrying its
  server-owned threshold, passed through unchanged.
* ``gate``     -> ``POST /v1/gate {target, dataset_id, bundle}`` -> ``{"passed": bool}``.

Metric selection is by the registered bundle name (the service owns the metric set + per-bundle
thresholds); no bare metric names go on the wire, so the service's fail-closed unknown-metric
rejection can never be triggered by this client. ``dataset_id`` is sent at both the top level
and inside ``target`` because the service 422s if they diverge.

Kept dependency-light: the ``Authorization`` / signed-actor headers are injectable
(``auth_headers=``), so a consumer can pass ``hex_service_kit.s2s.client_headers()`` without this
package depending on it. The base-URL https guard is
:func:`~agent_eval_kit._urls.require_secure_url`, shared with the opt-in local-model judge so
the transport rule has one home rather than one copy per client.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import PurePath

import httpx

from ._urls import require_secure_url
from .report import EvalMetricResult, EvalReport

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
#: Prompt/agent version tag; bump when the prompt corpus changes, or source it from a registry.
_DEFAULT_PROMPT_VERSION = "v1"

AuthHeaders = Mapping[str, str] | Callable[[], Mapping[str, str]]


def _assurance_passed(quality: bool, attested: bool, redteam: bool) -> bool:
    return quality and attested and redteam


class GateClientError(RuntimeError):
    """Raised when the gate service returns a non-2xx response or is unreachable."""


class PromotionGateClient:
    """HTTP client for a promotion-gate ("AI quality" / model-risk) service."""

    def __init__(
        self,
        base_url: str,
        *,
        bundle: str,
        model: str,
        prompt_version: str = _DEFAULT_PROMPT_VERSION,
        auth_headers: AuthHeaders | None = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = require_secure_url(base_url, service=type(self).__name__, kind="gate base")
        self._bundle = bundle
        self._model = model
        self._prompt_version = prompt_version
        self._auth_headers = auth_headers
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        source = self._auth_headers
        if source is None:
            return {}
        resolved = source() if callable(source) else source
        return dict(resolved)

    def _request_body(self, dataset_path: str) -> dict[str, object]:
        """Build the structured request body from a dataset path.

        ``dataset_id`` is the dataset file's basename without ``.jsonl`` and is sent at both the
        top level (the service loads the data from it) and inside ``target`` (the service requires
        them equal; a divergence is a 422).
        """
        dataset_id = PurePath(dataset_path).name.removesuffix(".jsonl")
        return {
            "target": {
                "model": self._model,
                "prompt_version": self._prompt_version,
                "dataset_id": dataset_id,
                "system": "",
            },
            "dataset_id": dataset_id,
            "bundle": self._bundle,
        }

    def evaluate(self, dataset_path: str) -> EvalReport:
        """Score ``dataset_path`` via the gate service and return an :class:`EvalReport`."""
        url = f"{self._base_url}/v1/evaluations"
        try:
            response = httpx.post(
                url,
                json=self._request_body(dataset_path),
                timeout=self._timeout,
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise GateClientError(f"gate request to {url} failed: {exc}") from exc
        if response.status_code // 100 != 2:
            raise GateClientError(
                f"gate service {url} returned {response.status_code}: {response.text[:500]}"
            )
        return self._parse(dataset_path, response.json())

    def gate(self, target: str) -> bool:
        """Promotion gate: True iff the service reports ``target`` passes.

        POSTs ``/v1/gate`` (not GET) so a missing dataset is a hard error, never a silent
        ``{"passed": false}`` indistinguishable from a real FAIL.
        """
        url = f"{self._base_url}/v1/gate"
        try:
            response = httpx.post(
                url,
                json=self._request_body(target),
                timeout=self._timeout,
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise GateClientError(f"gate request to {url} failed: {exc}") from exc
        if response.status_code // 100 != 2:
            raise GateClientError(
                f"gate service {url} returned {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise GateClientError("gate service returned non-JSON promotion evidence") from exc
        return self._parse_gate(target, body)

    @classmethod
    def _parse_gate(cls, dataset_path: str, body: object) -> bool:
        """Validate the complete Hrz4 GateDecision, never a naked aggregate boolean."""
        if not isinstance(body, dict):
            raise GateClientError("gate service promotion response must be a JSON object")
        raw_passed = body.get("passed")
        if raw_passed is True:
            aggregate_passed = True
        elif raw_passed is False:
            aggregate_passed = False
        else:
            raise GateClientError("gate service response must contain a boolean 'passed' field")

        eval_body = body.get("eval_report")
        if not isinstance(eval_body, dict):
            raise GateClientError("gate service response has no evaluation evidence")
        report = cls._parse(dataset_path, eval_body)
        if not report.attested:
            raise GateClientError("gate service evaluation evidence is not attested")
        required_eval = {
            "run_id": report.run_id,
            "dataset_digest": report.dataset_digest,
            "evaluator": report.evaluator,
        }
        missing_eval = [name for name, value in required_eval.items() if not value.strip()]
        if missing_eval or not report.artifact_refs:
            raise GateClientError(
                "gate service evaluation evidence lacks durable identifiers or artifact refs"
            )

        redteam = body.get("redteam_report")
        if not isinstance(redteam, dict):
            raise GateClientError("gate service response has no red-team evidence")
        rows = redteam.get("results")
        if not isinstance(rows, list) or not rows:
            raise GateClientError("gate service response has no red-team results")
        redteam_passed = True
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateClientError(f"red-team result {index} has no boolean verdict")
            row_passed = row.get("passed")
            if type(row_passed) is not bool:
                raise GateClientError(f"red-team result {index} has no boolean verdict")
            blocked = row.get("blocked")
            if type(blocked) is not bool or row_passed is not blocked:
                raise GateClientError(f"red-team result {index} has contradictory evidence")
            redteam_passed = redteam_passed and bool(row_passed)
        if type(redteam.get("passed")) is not bool or redteam["passed"] is not redteam_passed:
            raise GateClientError("red-team aggregate verdict contradicts its evidence")

        model_card_ref = body.get("model_card_ref")
        mrm_ref = body.get("mrm_evidence_ref")
        if not isinstance(model_card_ref, str) or not model_card_ref.strip():
            raise GateClientError("gate service response has no durable model-card reference")
        if not isinstance(mrm_ref, str) or not mrm_ref.strip():
            raise GateClientError("gate service response has no durable MRM evidence reference")

        derived = _assurance_passed(report.passed, report.attested, redteam_passed)
        if aggregate_passed is not derived:
            raise GateClientError("gate service aggregate verdict contradicts assurance evidence")
        return derived

    @staticmethod
    def _parse(dataset_path: str, body: dict[str, object]) -> EvalReport:
        # The service returns per-metric rows under "results" (NOT "metrics"); each row carries the
        # server-owned per-bundle threshold, which we pass through unchanged.
        if not isinstance(body, dict):
            raise GateClientError("gate service evaluation response must be a JSON object")
        rows = body.get("results")
        if not isinstance(rows, list) or not rows:
            raise GateClientError("gate service evaluation response has no metric results")
        n_examples = body.get("n_examples")
        if type(n_examples) is not int or n_examples <= 0:
            raise GateClientError("gate service evaluation response has no evaluated examples")

        parsed: list[EvalMetricResult] = []
        seen: set[str] = set()
        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                raise GateClientError(f"metric result {index} is not a JSON object")
            metric = item.get("metric")
            score = item.get("score")
            threshold = item.get("threshold")
            passed = item.get("passed")
            if not isinstance(metric, str) or not metric.strip() or metric in seen:
                raise GateClientError(f"metric result {index} has a missing or duplicate name")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
            ):
                raise GateClientError(
                    f"metric result {metric!r} has a non-numeric score or threshold"
                )
            if not isfinite(float(score)) or not isfinite(float(threshold)):
                raise GateClientError(f"metric result {metric!r} has non-finite evidence")
            if not isinstance(passed, bool):
                raise GateClientError(f"metric result {metric!r} has no boolean verdict")
            derived = float(score) >= float(threshold)
            if passed is not derived:
                raise GateClientError(
                    f"metric result {metric!r} verdict contradicts its score and threshold"
                )
            seen.add(metric)
            parsed.append(
                EvalMetricResult(
                    metric=metric,
                    score=float(score),
                    threshold=float(threshold),
                    passed=passed,
                )
            )

        report = EvalReport(dataset=dataset_path, results=tuple(parsed), n_examples=n_examples)
        server_passed = body.get("passed")
        if server_passed is not None and (
            type(server_passed) is not bool or server_passed is not report.passed
        ):
            raise GateClientError("gate service aggregate verdict contradicts metric evidence")
        artifact_refs_raw = body.get("artifact_refs", [])
        if not isinstance(artifact_refs_raw, list) or not all(
            isinstance(item, str) and bool(item.strip()) for item in artifact_refs_raw
        ):
            raise GateClientError("gate service artifact_refs must contain nonblank strings")
        attested = body.get("attested", False)
        if not isinstance(attested, bool):
            raise GateClientError("gate service attested field must be boolean")
        string_fields: dict[str, str] = {}
        for name, default in (
            ("run_id", ""),
            ("dataset_version", "v1"),
            ("dataset_digest", ""),
            ("evaluator", ""),
            ("schema_version", "eval-run/v1"),
        ):
            value = body.get(name, default)
            if not isinstance(value, str) or not value.strip():
                raise GateClientError(f"gate service {name} must be a nonblank string")
            string_fields[name] = value
        optional_fields: dict[str, str | None] = {}
        for name in ("trace_id", "correlation_id"):
            value = body.get(name)
            if value is None:
                optional_fields[name] = None
            elif isinstance(value, str) and value.strip():
                optional_fields[name] = value
            else:
                raise GateClientError(f"gate service {name} must be null or a nonblank string")
        return EvalReport(
            dataset=report.dataset,
            results=report.results,
            n_examples=report.n_examples,
            run_id=string_fields["run_id"],
            dataset_version=string_fields["dataset_version"],
            dataset_digest=string_fields["dataset_digest"],
            evaluator=string_fields["evaluator"],
            schema_version=string_fields["schema_version"],
            trace_id=optional_fields["trace_id"],
            correlation_id=optional_fields["correlation_id"],
            artifact_refs=tuple(artifact_refs_raw),
            attested=attested,
        )
