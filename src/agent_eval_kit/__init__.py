"""agent-eval-kit: the shared evaluation scaffold for hexagonal agent repos.

One versioned source of truth for the evaluation layer a service re-implements:

* **The report types** (:mod:`agent_eval_kit.report`) - ``EvalMetricResult`` / ``EvalReport``
  and their console rendering, the shape both evaluators speak.
* **The ``--mode smoke|gate`` scaffold** (:mod:`agent_eval_kit.modes`) - ``eval_main`` gives a
  project the standard CLI, aligned report output, and the fail-closed exit codes; the project
  supplies only its offline scorer and its gate runner.
* **The promotion-gate client** (:mod:`agent_eval_kit.gate_client`) - one HTTP contract for a
  promotion-gate service (structured target, registered bundle, ``results[]`` parse, POST gate).
* **The not-falsely-green harness** (:mod:`agent_eval_kit.harness`) - ``assert_can_go_red`` turns
  "prove this metric can fail" into a one-liner, the guard against a metric that cannot go red.
* **The offline judge harness** (:mod:`agent_eval_kit.judge`) - grade a model's NARRATIVE against
  reference criteria with no model at all by default, or with a locally served one by explicit
  configuration, and prove the judge itself can go red.
* **The per-vertical quality floors** (:mod:`agent_eval_kit.floors`) - the named number, held as
  data, below which a reduced (laptop / on-prem) profile is UNFIT for a vertical rather than
  merely degraded.
* **The reviewed thresholds** (:mod:`agent_eval_kit.rubrics`) - read a metric's bar out of a
  rubric file that carries the argument for it, and fail the build both when a metric has no
  reviewed bar AND when a bar names no metric.
* **The golden-set loader** (:mod:`agent_eval_kit.datasets`) - comment-tolerant JSONL, a content
  digest, and refusal of the three shapes that make a gate report a number over nothing.
* **The judged narrative run** (:mod:`agent_eval_kit.narrative`) - the whole
  criteria-plus-floors-plus-expectation-table runner as one call.
* **The replay** (:mod:`agent_eval_kit.replay`, :mod:`agent_eval_kit.recording`) - score the
  rubrics against a REAL model's committed, scrubbed words, offline, and refuse a whole
  recording batch on one leak rather than writing a scrubbed lie.
* **Retrieval quality** (:mod:`agent_eval_kit.retrieval`) - recall@k, precision@k and MRR over
  labelled query/passage pairs, measured upstream of what the model did with them.
* **The denominator rule** (:mod:`agent_eval_kit.denominators`) - a 0.90 bar over seven
  positives is a 1.0 wearing a 0.90 label, and this is the assertion that says so.
* **The render-and-check golden set** (:mod:`agent_eval_kit.golden`) - inputs rendered from the
  shipped demo data, expectations hand-written, staleness a build failure.

The report/modes/harness/ports/judge/floors layers are pure stdlib; the gate client needs
``httpx``. Kept dependency-light: the gate client takes injectable auth headers so a consumer can
pass ``hex_service_kit.s2s.client_headers()`` without a hard dependency.

Two modules are deliberately NOT imported eagerly here, so this package imports with no HTTP
client installed: :mod:`agent_eval_kit.gate_client` (the promotion-gate client) and
:mod:`agent_eval_kit.local_model_judge` (the opt-in model-backed judge). They, and the
``PromotionGateClient`` / ``GateClientError`` / ``LocalModelJudge`` names, are resolved LAZILY on
first attribute access (PEP 562), so ``from agent_eval_kit import PromotionGateClient`` still
works unchanged while merely importing the package does not pull ``httpx`` in. This matters
because every consuming repo's DOMAIN layer reaches this package - ``ports/observability.py``
does ``from agent_eval_kit import EvaluationGatePort``, and several ``domain/`` modules import
from :mod:`agent_eval_kit.report` - and importing any submodule executes this file first. An
eager ``from . import gate_client`` therefore put ``httpx`` in the import graph of every decision
core in the fleet. ``tests/test_core_purity.py`` is the guard that keeps it out.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import (
    datasets,
    denominators,
    floors,
    golden,
    harness,
    judge,
    modes,
    narrative,
    recording,
    replay,
    report,
    retrieval,
    rubrics,
)
from .datasets import DatasetError, dataset_digest, fraction, load_jsonl, warn_unmeasured
from .denominators import (
    DenominatorError,
    assert_denominator_supports,
    assert_each_denominator_supports,
    required_positives,
)
from .floors import (
    FLOORS_SCHEMA,
    Fitness,
    FitnessVerdict,
    FloorDataError,
    MissingFloorError,
    QualityFloor,
    QualityFloorError,
    QualityFloors,
    load_quality_floors,
)
from .golden import GoldenRenderError, render_main
from .harness import (
    NotFalselyGreenError,
    assert_can_go_red,
    assert_each_can_go_red,
    prove_before_scoring,
)
from .judge import (
    CriterionScore,
    DeterministicNarrativeJudge,
    JudgeConfigError,
    JudgePort,
    JudgeRequest,
    JudgeSelection,
    JudgeUnavailableError,
    JudgeVerdict,
    NarrativeCriterion,
    assert_judge_can_go_red,
    build_judge,
)
from .modes import GateRunner, SmokeRunner, build_parser, eval_main
from .narrative import Measured, NarrativeEvalError, narrative_main
from .ports import EvaluationGatePort
from .recording import Recorded, RecordingRefused, write_recordings
from .replay import ReplayAdapter, ReplayError, recording_key
from .report import EvalMetricResult, EvalReport, print_report
from .retrieval import RetrievalCase, RetrievalError, RetrievalScores, score_retrieval
from .rubrics import Rubric, RubricError, Rubrics, load_rubrics

if TYPE_CHECKING:  # Type checkers resolve the deferred names statically; the runtime defers them.
    from . import gate_client, local_model_judge
    from .gate_client import GateClientError, PromotionGateClient
    from .local_model_judge import LocalModelJudge

__version__ = "0.0.3"

#: The names served by :func:`__getattr__`, mapped to the submodule each one lives in. Both
#: submodules speak HTTP, and neither may be in the import graph of a consumer's decision core.
_DEFERRED_NAMES: dict[str, str] = {
    "gate_client": "gate_client",
    "GateClientError": "gate_client",
    "PromotionGateClient": "gate_client",
    "local_model_judge": "local_model_judge",
    "LocalModelJudge": "local_model_judge",
}


def __getattr__(name: str) -> Any:
    """Resolve the HTTP-speaking surface on first access (PEP 562), not at package import.

    ``import_module`` rather than ``from . import gate_client``: the ``from``-import form runs
    ``importlib._bootstrap._handle_fromlist``, which probes the parent package with ``hasattr``
    before importing the submodule, and that call re-enters THIS function - an infinite recursion
    that only shows up in a process where nothing has already bound the submodule. ``import_module``
    goes straight to the import machinery and still binds the submodule onto this package, so
    every later access is an ordinary attribute lookup and this runs at most once.
    """
    module_name = _DEFERRED_NAMES.get(name)
    if module_name is not None:
        module = import_module(f"{__name__}.{module_name}")
        return module if name == module_name else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep the deferred names discoverable by ``dir()`` and by tab completion."""
    return sorted(set(globals()) | set(_DEFERRED_NAMES))


__all__ = [
    "FLOORS_SCHEMA",
    "CriterionScore",
    "DatasetError",
    "DenominatorError",
    "GoldenRenderError",
    "Measured",
    "NarrativeEvalError",
    "Recorded",
    "RecordingRefused",
    "ReplayAdapter",
    "ReplayError",
    "RetrievalCase",
    "RetrievalError",
    "RetrievalScores",
    "Rubric",
    "RubricError",
    "Rubrics",
    "DeterministicNarrativeJudge",
    "EvaluationGatePort",
    "EvalMetricResult",
    "EvalReport",
    "Fitness",
    "FitnessVerdict",
    "FloorDataError",
    "GateClientError",
    "GateRunner",
    "JudgeConfigError",
    "JudgePort",
    "JudgeRequest",
    "JudgeSelection",
    "JudgeUnavailableError",
    "JudgeVerdict",
    "LocalModelJudge",
    "MissingFloorError",
    "NarrativeCriterion",
    "PromotionGateClient",
    "NotFalselyGreenError",
    "QualityFloor",
    "QualityFloorError",
    "QualityFloors",
    "SmokeRunner",
    "__version__",
    "assert_can_go_red",
    "assert_denominator_supports",
    "assert_each_can_go_red",
    "assert_each_denominator_supports",
    "assert_judge_can_go_red",
    "build_judge",
    "build_parser",
    "dataset_digest",
    "datasets",
    "denominators",
    "eval_main",
    "floors",
    "fraction",
    "gate_client",
    "golden",
    "harness",
    "judge",
    "load_jsonl",
    "load_quality_floors",
    "load_rubrics",
    "local_model_judge",
    "modes",
    "narrative",
    "narrative_main",
    "print_report",
    "prove_before_scoring",
    "recording",
    "recording_key",
    "render_main",
    "replay",
    "report",
    "required_positives",
    "retrieval",
    "rubrics",
    "score_retrieval",
    "warn_unmeasured",
    "write_recordings",
]
