# agent-eval-kit

The shared **evaluation scaffold** for hexagonal (ports-and-adapters) service repos. One versioned
source of truth for the eval layer applications re-implement: the report types, the `run_eval.py
--mode smoke|gate` CLI, a promotion-gate HTTP client, a harness that makes "prove this metric
can go red" a one-liner, and an offline narrative-quality judge with named per-vertical floors.

**Pure standard library except the two HTTP clients** (which need `httpx`) - and importing the
package imports neither, so `httpx` never lands in a consumer's decision core. See
[Design invariants](#design-invariants-do-not-fix-these).

## Why it exists

In a polyrepo, copy-paste is the only sharing mechanism, so the `--mode` split and the gate-client
contract get pasted into every service and drift. Worse, an eval metric can quietly become unable
to fail (a scorer reading its own output, a golden set that planted no target), so it is a constant
1.0 that proves nothing. This package retires both: adopt the scaffold and the harness by a version
bump.

The judge and the floors close the third gap. Every eval in the fleet scored a deterministic core
against a deterministic fake model, so nothing measured model NARRATIVE quality, and the profile
running the weaker local model was the one profile with no quality measurement at all: quality was
scored by a managed authority the reduced profile cannot reach. The judge measures narrative
quality where the weak model runs, and the floor says how good is good enough, per vertical.

## What you get

```python
from pathlib import Path
from agent_eval_kit import eval_main, EvalReport, EvalMetricResult, PromotionGateClient, assert_can_go_red

# 1) The --mode smoke|gate CLI. Supply your offline scorer and your gate runner; get the
#    standard CLI, aligned output, and fail-closed exit codes.
def smoke(dataset: Path) -> EvalReport:
    ...  # your deterministic heuristic evaluator
    return EvalReport(dataset=str(dataset), results=(EvalMetricResult.scored("pii_safety", 1.0, 0.99),))

def gate(dataset: Path) -> tuple[EvalReport, bool]:
    client = PromotionGateClient("https://quality.internal", bundle="example-bundle", model="my-model")
    return client.evaluate(str(dataset)), client.gate(str(dataset))

if __name__ == "__main__":
    raise SystemExit(eval_main(smoke=smoke, gate=gate, default_dataset=Path("eval/datasets/golden.jsonl")))
```

```python
# 2) Prove a metric is not structurally falsely green: it PASSES clean and FAILS degraded.
assert_can_go_red(
    my_pii_scorer,
    green="fully redacted output",
    red="applicant NRIC S1234567D leaked",   # obviously fictional
    threshold=0.99,
    metric="pii_safety",
)
```

```python
# 3) Measure NARRATIVE quality offline, and compare it to a named per-vertical floor.
from agent_eval_kit import (
    NarrativeCriterion, JudgeRequest, JudgeSelection, build_judge, load_quality_floors,
)

criterion = NarrativeCriterion(
    name="explains_the_escalation",
    must_cover=("the source of wealth is unevidenced",),
    must_cite=("KYC-POL-4.2",),          # obviously fictional
    must_not_say=("guaranteed clean",),
)
judge = build_judge(JudgeSelection.from_env())          # unset env => deterministic, offline
verdict = judge.grade(JudgeRequest(candidate=model_output, criteria=(criterion,)))

floors = load_quality_floors("config/quality-floors.toml")
fitness = floors.assess_verdict("doc1-cdd-sow", verdict, profile="reduced")
# fitness.fitness is FIT / DEGRADED / UNFIT; fitness.as_metric_result() drops into an EvalReport.
```

The gate client speaks a hardened contract: a structured `target`, top-level `dataset_id` equal to
`target.dataset_id`, selection by the registered `bundle` (no bare metric names on the wire),
response parsed from `results[]`, and a POST promotion gate. Auth headers are injectable, so a
consumer passes `hex_service_kit.s2s.client_headers()` for signed S2S calls without this package
depending on `hex-service-kit`.

## Modules

| Module | What it owns | Deps |
|---|---|---|
| `report` | `EvalMetricResult`, `EvalReport`, `print_report` | stdlib |
| `modes` | `eval_main`, `build_parser` (`--mode smoke\|gate`, fail-closed exit codes) | stdlib |
| `gate_client` | `PromotionGateClient` (`evaluate` / `gate`), `GateClientError` | `httpx`, imported lazily |
| `harness` | `assert_can_go_red`, `assert_each_can_go_red`, `NotFalselyGreenError` | stdlib |
| `judge` | `NarrativeCriterion`, `JudgeRequest`, `JudgeVerdict`, `DeterministicNarrativeJudge`, `JudgeSelection`, `build_judge`, `assert_judge_can_go_red` | stdlib |
| `floors` | `QualityFloors`, `QualityFloor`, `Fitness`, `FitnessVerdict`, `load_quality_floors` | stdlib |
| `local_model_judge` | `LocalModelJudge`, the opt-in chat-completions judge | `httpx`, imported per call |
| `ports` | `EvaluationGatePort`, the evaluation-gate Protocol | stdlib |

## The quality floor

The judge answers "how good is this narrative"; the floor answers "good enough for what". Floors
are **data**, in a TOML or JSON file a model-risk function owns:

```toml
schema = "quality-floors/v1"

[verticals.doc1-cdd-sow]
floor  = 0.70   # below this, a profile is UNFIT for this vertical, not merely degraded
target = 0.88   # at or above this, it is FIT. Between the two it is DEGRADED.
```

Which is what turns "a reduced profile serves visibly lower narrative quality" from an adjective
into a measurement a gate can act on. Every path fails closed: an unnamed vertical raises rather
than inheriting a bar, an empty floor table refuses to load, a floor of zero is refused because
it could never refuse anything, and a verdict with no graded criteria is UNFIT rather than a
pass.

## Choosing the judge

| `AGENT_EVAL_JUDGE` | Judge | Needs |
|---|---|---|
| unset (the default) | `DeterministicNarrativeJudge` | nothing: no server, no network, no credentials |
| `deterministic` | the same, named explicitly | nothing |
| `local-model` | `LocalModelJudge` | `AGENT_EVAL_JUDGE_BASE_URL` and `AGENT_EVAL_JUDGE_MODEL`, both required |

Set but empty is a configuration error rather than a silent fall back to the default. The
model-backed judge speaks OpenAI-compatible chat completions with NO `/v1` prefix, which is what
a local MLX server serves; pass `path="/v1/chat/completions"` for a server that uses the
conventional prefix.

## Install

```sh
pip install agent-eval-kit
```

## Develop

```sh
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests && mypy src && pytest
```

The hard gate is ruff (lint) + ruff (format check, ruff pinned exactly) + mypy `--strict` (src
only) + pytest, on Python 3.12 and 3.13. `respx` mocks the gate endpoints so the contract test
never talks to a real service.

## Design invariants (do not "fix" these)

- **Fail closed in gate mode.** `eval_main` exits 0 only when BOTH the scored report passes and
  the authority's gate verdict is True, so an offline smoke result can never be relabelled a
  promotion pass.
- **The gate is a POST, not a GET.** A missing dataset is a hard error, never a silent
  `{"passed": false}` indistinguishable from a real FAIL.
- **Selection is by bundle, never bare metric names.** The service owns the metric set +
  per-bundle thresholds; the client passes them through unchanged.
- **Dependency-light.** Auth headers are injectable rather than a hard dependency on
  `hex-service-kit`, so this package installs and tests standalone.
- **The judge default is offline, and a model is opt-in.** A gate has to run with no model
  server, no network and no credentials, because that is the fleet's portability proof. So an
  unset `AGENT_EVAL_JUDGE` selects the deterministic judge, and no test in this package talks to
  a server: `tests/test_local_model_judge.py` mocks the endpoint with `respx`.
- **A judge that cannot answer raises.** Never a default score. A number invented by a failure
  is indistinguishable from a measurement and gets read as one.
- **An absent measurement is the weakest evidence, never the strongest.** `JudgeVerdict` with no
  scores has `score == 0.0` and `meets(floor)` False, and `QualityFloors.assess_verdict` calls
  it UNFIT. This is the `EvalReport(metrics=[]).passed is True` defect, which this fleet has
  already paid for once, closed by construction.
- **Importing the package must not import `httpx`.** Every consuming repo's domain layer reaches
  this package (`ports/observability.py` does `from agent_eval_kit import EvaluationGatePort`), and
  importing any submodule executes `__init__.py` first, so whatever that file imports eagerly is in
  the decision core's import graph. `gate_client` is therefore resolved lazily on first attribute
  access (PEP 562): `from agent_eval_kit import PromotionGateClient` still works, importing the
  package does not pull an HTTP client in. `tests/test_core_purity.py` runs the import with
  `httpx` blocked at `sys.meta_path` and is what keeps this true.

## License

Apache-2.0.
