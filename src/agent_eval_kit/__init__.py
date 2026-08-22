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

from . import floors, harness, judge, modes, report
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
from .harness import NotFalselyGreenError, assert_can_go_red, assert_each_can_go_red
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
from .ports import EvaluationGatePort
from .report import EvalMetricResult, EvalReport, print_report

if TYPE_CHECKING:  # Type checkers resolve the deferred names statically; the runtime defers them.
    from . import gate_client, local_model_judge
    from .gate_client import GateClientError, PromotionGateClient
    from .local_model_judge import LocalModelJudge

__version__ = "0.0.1"

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
    "assert_each_can_go_red",
    "assert_judge_can_go_red",
    "build_judge",
    "build_parser",
    "eval_main",
    "floors",
    "gate_client",
    "harness",
    "judge",
    "load_quality_floors",
    "local_model_judge",
    "modes",
    "print_report",
    "report",
]
