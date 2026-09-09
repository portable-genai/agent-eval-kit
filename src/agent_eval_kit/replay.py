"""Score the rubrics against a REAL model's words, offline, by replaying recorded output.

Every deterministic metric in this fleet scores a template adapter that structurally cannot
invent a figure. That makes the citation and grounding metrics a measurement of the VALIDATOR
rather than of a model's restraint: they stay green through a model swap, a prompt regression, a
temperature change or a context-window truncation, because no model is in the measured path at
all.

The fix cannot be "call the model in the gate": the gate must pass with no network, no
credentials and no cloud SDK, and that property is not for trading. So a real call is made ONCE,
by hand, by an authoring script (see :mod:`agent_eval_kit.recording`), its output is scrubbed and
committed, and this module replays it. The same rubrics and the same hand-written labels then
score real model text with nothing reachable.

Eval-only, and deliberately not a product adapter. A replay adapter belongs beside the eval, not
under ``src/``: nothing a deployment binds should be able to serve pre-recorded answers to a
customer.

Fails closed in the one way that matters. A prompt with no recording RAISES, naming the case. It
never falls back to a template drafter and it never reaches the network, because a replay that
quietly substituted a different drafter would report a score for a model that produced none of
it.

One subtlety keeps that honest end to end, and it is the reason :attr:`ReplayAdapter.MISSES`
exists. A well-built pipeline treats ANY generation failure as silence, deliberately: for the
product, a model outage must degrade to "no suggestion", never to an unvalidated fallback. That
product property swallows this adapter's raise, and silence is scoreable, so a stale recording
would grade as a model that declined everything and would PASS wherever silence was the expected
answer. So every miss is also recorded on the class, and a runner fails the whole replay run
when the list is non-empty, whatever the metrics say.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "ReplayAdapter",
    "ReplayError",
    "load_recordings",
    "recording_key",
]


class ReplayError(RuntimeError):
    """Raised when a recording is absent, unreadable, empty, or stale for the case at hand."""


def recording_key(model: str, prompt: str, inputs: Iterable[str] = ()) -> str:
    """The identity of one generation call: the model, the prompt, and what it was given.

    All three, because all three change the answer. Keying on the prompt alone would replay one
    market's recorded reply for another market's retrieved passages and call it evidence.

    ``inputs`` are sorted before hashing, so a retriever that returns the same evidence in a
    different order does not invalidate a recording. What was retrieved is the input; the order
    it arrived in is not.
    """
    joined = ",".join(sorted(str(item) for item in inputs))
    return hashlib.sha256(f"{model}\n{prompt}\n{joined}".encode()).hexdigest()


def load_recordings(path: Path) -> dict[str, dict[str, Any]]:
    """Read a committed recording file, keyed by :func:`recording_key`. Comment-tolerant."""
    if not path.exists():
        raise ReplayError(
            f"no recorded model output at {path}. Record it once against the managed profile "
            "with this repository's recording script, review the scrubbed result, and commit "
            "it. Until then the rubrics score the offline template drafter, which is a "
            "measurement of the validator rather than of a model's restraint."
        )
    rows: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path}:{number}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("key"):
            raise ReplayError(f"{path}:{number}: every recording must carry a 'key'")
        rows[str(row["key"])] = row
    if not rows:
        raise ReplayError(f"{path} contains no recordings, so nothing can be replayed")
    return rows


class ReplayAdapter:
    """Base for an eval-only generation adapter that replays committed managed-model output.

    A subclass supplies the model name and the fixture path, and calls :meth:`replay` from
    whatever method its repo's generation port declares. Everything that makes the replay
    trustworthy lives here: the key, the fail-closed miss, and the class-level miss list.
    """

    #: Every key that had no recording, across all instances of a run. Class-level because a
    #: container constructs the instance and the runner never holds it, and because the raise
    #: below is converted into silence by a kernel that treats generation failure as "no
    #: suggestion". Silence is scoreable, so the raise alone is not enough: a runner must clear
    #: this list before a replay run and fail the run if anything lands in it.
    MISSES: ClassVar[list[str]] = []

    def __init__(self, model: str, fixture: Path) -> None:
        self._model = model
        self._fixture = Path(fixture)
        self._rows: dict[str, dict[str, Any]] | None = None

    @classmethod
    def reset_misses(cls) -> None:
        """Clear the miss list before a replay run. Call it, or a previous run's misses count."""
        cls.MISSES.clear()

    @classmethod
    def assert_no_misses(cls) -> None:
        """Fail the whole replay run when any case had no recording, whatever the metrics say."""
        if cls.MISSES:
            raise ReplayError(
                f"{len(cls.MISSES)} case(s) had no recorded model output, so the run scored "
                "something other than the model it claims to have scored. The pipeline converts "
                "a generation failure into silence by design, and silence passes wherever "
                "silence was the expected answer, so this is checked separately from the "
                f"metrics. Re-record. Missing keys: {[key[:12] for key in cls.MISSES]}"
            )

    def recordings(self) -> dict[str, dict[str, Any]]:
        if self._rows is None:
            self._rows = load_recordings(self._fixture)
        return self._rows

    def key_for(self, prompt: str, inputs: Sequence[str] = ()) -> str:
        return recording_key(self._model, prompt, inputs)

    def replay(self, prompt: str, inputs: Sequence[str] = ()) -> Mapping[str, Any] | None:
        """The recorded reply for this call, or raise. Never a fallback, never the network.

        A recorded ``null`` is returned as ``None`` rather than raised: the model declined, the
        pipeline treats that as no output, and replaying it as an error would hide a case worth
        scoring.
        """
        key = self.key_for(prompt, inputs)
        try:
            rows = self.recordings()
        except ReplayError:
            # An unreadable or empty fixture is a miss for every call, not only for this key.
            type(self).MISSES.append(key)
            raise
        row = rows.get(key)
        if row is None:
            type(self).MISSES.append(key)
            raise ReplayError(
                f"no recorded model output for {key[:12]} (model {self._model!r}). The recording "
                "is out of date with the corpus or the prompt. Re-record rather than falling "
                "back: a replay that substituted another drafter would report a score for a "
                "model that produced none of it."
            )
        response = row.get("response")
        return response if isinstance(response, Mapping) else None
