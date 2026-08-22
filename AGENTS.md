# agent-eval-kit

The shared working agreement is [`.github/AGENTS.md`](https://github.com/portable-genai/.github/blob/main/AGENTS.md).
It carries the architecture rules, the gate contract, the fleet invariants, the
falsification discipline, versions and house style, and it holds in every repository
here. Read it first. This file carries only what is specific to this one.

## What this is

`agent-eval-kit` is the **evaluation scaffold** for hexagonal (ports-and-adapters) services,
packaged once: the report types, the `run_eval.py --mode smoke|gate` CLI, a promotion-gate client,
a not-falsely-green harness, and an offline narrative-quality judge with named per-vertical
quality floors. An application supplies its own scorers, criteria and floor numbers; this package
supplies the structure around them.

## Commands

A venv exists at `.venv`. Setup from scratch:

```sh
pip install -e ".[dev]"        # httpx + ruff (pinned) + mypy + pytest + respx
```

The full CI gate, in order (all four must pass):

```sh
ruff check src tests
ruff format --check src tests   # ruff pinned EXACTLY in pyproject.toml so formatting never drifts
mypy src                        # strict; src only
pytest                          # -q, testpaths=tests
```

Run a single test:

```sh
pytest tests/test_gate_client.py -q
pytest tests/test_harness.py -k falsely_green -q
```

## Hard constraints

- **Only the two HTTP clients have a runtime dependency (`httpx`).** `report`, `modes`, `harness`,
  `ports`, `judge`, `floors` and `_urls` are pure stdlib. Do not add a dependency to them.
- **Never import `gate_client` or `local_model_judge` eagerly from `__init__.py`.** Consumers'
  DOMAIN layers import this package, and importing any submodule executes `__init__.py` first, so
  an eager `from . import gate_client` puts `httpx` in every consumer's decision core. Both are
  resolved lazily via a module `__getattr__` (PEP 562); `tests/test_core_purity.py` fails if that
  is undone. Resolve them with `importlib.import_module`, never `from . import gate_client`: the
  `from` form probes the parent package with `hasattr`, which re-enters `__getattr__` and recurses
  forever.
- **`local_model_judge` imports `httpx` INSIDE the methods that make requests**, so even importing
  that module directly stays stdlib-only. Do not lift the import to module scope; a purity test
  fails if you do.
- **The judge default is offline and a model is opt-in.** An unset `AGENT_EVAL_JUDGE` selects
  `DeterministicNarrativeJudge`, `local-model` requires both the base URL and the model id, and
  set-but-empty is an error rather than a fall back. No test may depend on a model server being
  up; the wire contract is pinned offline with `respx`.
- **An absent measurement is never a pass.** A `JudgeVerdict` with no scores has `score == 0.0`,
  `meets()` False, and is UNFIT at the floor. A `JudgeRequest` with no criteria, and a
  `NarrativeCriterion` that requires nothing, are refused at construction. This is the
  `EvalReport(metrics=[]).passed is True` class of defect and it must stay closed.
- **Quality floors are DATA and are never defaulted.** An unnamed vertical raises
  `MissingFloorError`, an empty table refuses to load, and a floor of zero is refused.
- **`__version__` and the `pyproject.toml` version must be bumped together.** A release where the
  two disagreed has already caused real confusion; `tests/test_core_purity.py` now pins them.
- **Python >=3.12**, mypy `strict = true`, ruff line-length 100 with `E,F,I,UP,B,SIM`.
- **Fail closed.** `eval_main` gate mode exits 0 only when the scored report AND the authority's
  verdict both pass. The gate call is a POST (a missing dataset must be a hard error, never a
  silent `{"passed": false}`).
- **Dependency-light.** The gate client takes injectable `auth_headers` rather than importing
  `hex-service-kit`, so this package installs and tests standalone. Do not add a hard dependency on
  another package to the stdlib modules.

## Architecture

Eight modules in `src/agent_eval_kit/`, re-exported flat from `__init__.py` (`gate_client` and
`local_model_judge` lazily):

- **report.py** - `EvalMetricResult` / `EvalReport` + `print_report`.
- **modes.py** - `eval_main(smoke=..., gate=..., default_dataset=...)`. A project passes two
  callables; its own scorers stay in the application.
- **gate_client.py** - `PromotionGateClient`, an HTTP client for a promotion-gate service, with
  injectable `auth_headers`.
- **harness.py** - `assert_can_go_red` / `assert_each_can_go_red`, the guard against a metric that
  cannot go red.
- **judge.py** - the offline judge harness: `NarrativeCriterion` / `JudgeRequest` /
  `JudgeVerdict`, the `JudgePort` Protocol, `DeterministicNarrativeJudge` (the default),
  `JudgeSelection` / `build_judge` (the offline-by-default selection), and
  `assert_judge_can_go_red`, which turns the harness on the judge itself.
- **floors.py** - `QualityFloors` and `load_quality_floors`: the named per-vertical floor and
  target, loaded from a TOML or JSON file, and the FIT / DEGRADED / UNFIT verdict.
- **local_model_judge.py** - `LocalModelJudge`, the opt-in OpenAI-compatible chat-completions
  judge. Paths carry NO `/v1` prefix by default (the local MLX server's shape).
- **_urls.py** - `require_secure_url`, the https-or-loopback rule both HTTP clients obey.
- **ports.py** - `EvaluationGatePort`, the evaluation-gate Protocol every repo binds. Typing only.

The gate client speaks a specific wire contract (structured `target`, top-level `dataset_id`,
selection by `bundle`, `results[]` parse, POST `/v1/gate`); the respx test in
`tests/test_gate_client.py` pins that shape, so change the two together if the server contract
changes.
