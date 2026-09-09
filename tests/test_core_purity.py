"""The package must import with ``httpx`` absent, or it is not a decision-core dependency.

Every consuming repo's DOMAIN layer reaches this package: `ports/observability.py` does
`from agent_eval_kit import EvaluationGatePort`, and several `domain/` modules do
`from agent_eval_kit.report import EvalReport`. Both of those execute
`agent_eval_kit/__init__.py` first, so anything that file imports eagerly lands in the decision
core's import graph whether the core wanted it or not.

The package docstring states the invariant ("The report/modes/harness layers are pure stdlib; the
gate client needs ``httpx``"), and `hex-service-kit` states the discipline that keeps it true: two
modules are deliberately NOT re-exported from its `__init__`, so the kernel imports with neither a
web framework nor a tracer installed. `agent_eval_kit` had asserted the invariant and then broken
it six lines later with `from . import gate_client, ...`.

A prose invariant with no test is one refactor away from being undone, so this module is the
guard. It runs each check in a SUBPROCESS with a `sys.meta_path` finder that raises on the blocked
distributions: an in-process check would pass on nothing more than `httpx` already sitting in
`sys.modules` from another test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Blocked at the finder, before any loader runs, so a stale `sys.modules` entry cannot mask a
#: real import. `httpx` is this package's own runtime dependency; the rest are the heavyweight
#: adapter-layer distributions a consuming repo's decision core must equally import without.
_BLOCKED = ("httpx", "starlette", "fastapi", "opentelemetry", "google", "vertexai", "yaml")

_BLOCKER = f"""
import sys

BLOCKED = {_BLOCKED!r}


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"BLOCKED third-party import: {{fullname}}")
        return None


sys.meta_path.insert(0, _Blocker())
for _name in list(sys.modules):
    if _name.split(".")[0] in BLOCKED:
        del sys.modules[_name]
"""


def _run(body: str, *, blocked: bool) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a FRESH interpreter, optionally with the blocked distributions unimportable.

    Always a subprocess. In-process checks here would be worthless twice over: an import check
    would pass on nothing more than `httpx` already sitting in `sys.modules` from another test,
    and the lazy-resolution checks would pass on `agent_eval_kit.gate_client` having already been
    bound onto the package by `tests/test_gate_client.py`, which is exactly the state that hides a
    broken `__getattr__`.
    """
    prelude = _BLOCKER if blocked else ""
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _run_with_blocked_imports(body: str) -> subprocess.CompletedProcess[str]:
    return _run(body, blocked=True)


def test_the_package_imports_with_httpx_absent() -> None:
    """The load-bearing one: `from agent_eval_kit import EvaluationGatePort` in a domain layer."""
    result = _run_with_blocked_imports(
        """
        import agent_eval_kit

        from agent_eval_kit import EvalMetricResult, EvalReport, EvaluationGatePort, eval_main

        assert EvaluationGatePort is not None
        assert EvalReport is not None
        assert EvalMetricResult is not None
        assert eval_main is not None
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "importing agent_eval_kit dragged a blocked third-party package into the import graph "
        f"of every consumer's decision core:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_the_pure_submodules_import_with_httpx_absent() -> None:
    """Each stdlib-only submodule, imported directly; each executes the package `__init__` too."""
    result = _run_with_blocked_imports(
        """
        import agent_eval_kit.datasets
        import agent_eval_kit.denominators
        import agent_eval_kit.floors
        import agent_eval_kit.golden
        import agent_eval_kit.harness
        import agent_eval_kit.judge
        import agent_eval_kit.modes
        import agent_eval_kit.narrative
        import agent_eval_kit.ports
        import agent_eval_kit.recording
        import agent_eval_kit.replay
        import agent_eval_kit.report
        import agent_eval_kit.retrieval
        import agent_eval_kit.rubrics

        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_yaml_readers_import_and_run_their_non_yaml_paths_with_pyyaml_absent() -> None:
    """PyYAML is an EXTRA, so the two modules that can read YAML must not need it to load.

    A consumer's decision core reaches this package for its report types. A top-level
    `import yaml` in the rubric or dataset loader would put a parser in that import graph for
    the sake of a function no domain module calls, which is the same defect this package
    already fixed once for the HTTP client. The import lives inside the one branch that needs
    it, so every other format (.toml, .json, .jsonl) works with PyYAML uninstalled, and asking
    for YAML without it raises an error that says what to install.
    """
    result = _run_with_blocked_imports(
        """
        import json
        import tempfile
        from pathlib import Path

        NL = chr(10)

        from agent_eval_kit.datasets import load_jsonl
        from agent_eval_kit.rubrics import RubricError, load_rubrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "g.jsonl").write_text(NL.join(['# note', '{"id": "a", "kind": "routing"}', '']))
            assert len(load_jsonl(root / "g.jsonl", kinds={"routing"})) == 1

            rubrics = root / "rubrics"
            rubrics.mkdir()
            (rubrics / "a.toml").write_text(NL.join(['metric = "safety"', "threshold = 0.99", '']))
            (rubrics / "b.json").write_text(json.dumps({"metric": "recall", "threshold": 0.9}))
            assert load_rubrics(rubrics).thresholds() == {"safety": 0.99, "recall": 0.9}

            (rubrics / "c.yaml").write_text(NL.join(["metric: grounded", "threshold: 0.8", '']))
            try:
                load_rubrics(rubrics)
            except RubricError as exc:
                assert "PyYAML" in str(exc), exc
            else:
                raise AssertionError("a YAML rubric loaded with PyYAML blocked")

        print("OK")
        """
    )
    assert result.returncode == 0, (
        "the rubric or dataset loader needs PyYAML at import time, so every consumer's decision "
        f"core would gain a parser: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_the_new_offline_surface_runs_end_to_end_with_every_client_blocked() -> None:
    """The load-bearing claim of v0.0.2: all of it is a gate step, so none of it may reach out."""
    result = _run_with_blocked_imports(
        """
        from agent_eval_kit import (
            RetrievalCase,
            assert_denominator_supports,
            fraction,
            prove_before_scoring,
            recording_key,
            required_positives,
            score_retrieval,
        )
        from agent_eval_kit.denominators import DenominatorError

        assert required_positives(0.90) == 10
        assert fraction([]) == 0.0
        assert score_retrieval([RetrievalCase("q", ("a",), ("a",))], k=3).recall_at_k == 1.0
        assert len(recording_key("m", "p", ["a"])) == 64
        prove_before_scoring(lambda: None)
        try:
            assert_denominator_supports(0.90, 7, metric="m")
        except DenominatorError:
            pass
        else:
            raise AssertionError("the denominator rule did not refuse a thin corpus")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_offline_judge_harness_runs_with_httpx_absent() -> None:
    """The load-bearing claim of the judge harness: it measures quality with NO client at all.

    Importing the modules is not enough. The gate this defends runs the deterministic judge and
    scores its verdict against a floor, so the whole offline path is executed here in a process
    where an HTTP client cannot be imported.
    """
    result = _run_with_blocked_imports(
        """
        from agent_eval_kit import (
            DeterministicNarrativeJudge,
            JudgeRequest,
            JudgeSelection,
            NarrativeCriterion,
            QualityFloors,
            build_judge,
        )

        assert JudgeSelection.from_env({}).is_offline is True
        judge = build_judge()
        assert isinstance(judge, DeterministicNarrativeJudge)
        criterion = NarrativeCriterion(name="c", must_cover=("a nominee shareholder",))
        verdict = judge.grade(
            JudgeRequest(candidate="A nominee shareholder holds it.", criteria=(criterion,))
        )
        floors = QualityFloors.from_mapping(
            {"schema": "quality-floors/v1", "verticals": {"v": {"floor": 0.7, "target": 0.9}}}
        )
        assert floors.assess_verdict("v", verdict, profile="reduced").unfit is False
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "the offline judge path needs an HTTP client, so a gate could not run it with no model "
        f"server and no network:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_importing_the_package_does_not_import_the_gate_client() -> None:
    """The mutant guard: re-adding `gate_client` to the eager import turns this red.

    Without it, a future `from . import gate_client` could be re-introduced and stay green on
    any machine that happens to have `httpx` installed, which is every machine that installs
    this package.
    """
    result = _run_with_blocked_imports(
        """
        import sys

        import agent_eval_kit

        assert "agent_eval_kit.gate_client" not in sys.modules, (
            "agent_eval_kit.gate_client was imported eagerly by the package __init__"
        )
        assert "httpx" not in sys.modules
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_the_package_does_not_import_the_local_model_judge() -> None:
    """The second deferred module. The opt-in judge must cost a decision core nothing."""
    result = _run_with_blocked_imports(
        """
        import sys

        import agent_eval_kit

        assert "agent_eval_kit.local_model_judge" not in sys.modules, (
            "agent_eval_kit.local_model_judge was imported eagerly by the package __init__"
        )
        assert "httpx" not in sys.modules
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_local_model_judge_module_itself_imports_with_httpx_absent() -> None:
    """Even importing the adapter directly must not pull the client in: it imports per call.

    The deferred `__getattr__` alone would not be enough. A consumer that reaches for the
    adapter's module by name (to type-annotate it, or to build one behind its own flag) would
    otherwise drag `httpx` in at import time, which is the defect this package already fixed
    once for the gate client. Here the import is INSIDE the two methods that make requests, so
    the module is stdlib-only until a request is actually made.
    """
    result = _run_with_blocked_imports(
        """
        import sys

        import agent_eval_kit.local_model_judge as adapter

        assert "httpx" not in sys.modules
        judge = adapter.LocalModelJudge("http://127.0.0.1:8001", model="example/model")
        assert judge.name == "local-model:example/model"
        assert adapter.DEFAULT_CHAT_PATH == "/chat/completions"
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "importing the opt-in judge adapter pulled an HTTP client into the import graph:\n"
        f"{result.stderr}"
    )
    assert "OK" in result.stdout


def test_the_deferred_surface_is_still_reachable_from_the_package_root() -> None:
    """Deferring the import must not narrow the public surface: consumers import these by name.

    Repos across the fleet do `from agent_eval_kit import EvalReport, PromotionGateClient`, so the
    root-level names have to keep resolving; they are simply resolved on first ACCESS rather than
    at package import. Both access forms are exercised in a fresh interpreter, because the
    `from package import submodule` form is the one that recurses if `__getattr__` resolves the
    submodule with a `from`-import of its own.
    """
    result = _run(
        """
        import sys

        import agent_eval_kit

        # The from-import form, which routes through _handle_fromlist -> hasattr -> __getattr__.
        from agent_eval_kit import (
            GateClientError,
            LocalModelJudge,
            PromotionGateClient,
            gate_client,
            local_model_judge,
        )

        assert gate_client is sys.modules["agent_eval_kit.gate_client"]
        assert local_model_judge is sys.modules["agent_eval_kit.local_model_judge"]
        assert PromotionGateClient is gate_client.PromotionGateClient
        assert GateClientError is gate_client.GateClientError
        assert LocalModelJudge is local_model_judge.LocalModelJudge
        # Binding onto the package means the second access is a plain lookup, not __getattr__.
        assert agent_eval_kit.gate_client is gate_client
        assert agent_eval_kit.PromotionGateClient is PromotionGateClient
        assert agent_eval_kit.local_model_judge is local_model_judge

        names = {
            "gate_client",
            "GateClientError",
            "PromotionGateClient",
            "local_model_judge",
            "LocalModelJudge",
        }
        assert names <= set(agent_eval_kit.__all__)
        assert names <= set(dir(agent_eval_kit))
        print("OK")
        """,
        blocked=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "OK" in result.stdout


def test_the_attribute_access_form_also_resolves_in_a_fresh_interpreter() -> None:
    """Plain attribute access with nothing pre-bound: the other entry into `__getattr__`."""
    result = _run(
        """
        import agent_eval_kit

        client = agent_eval_kit.PromotionGateClient
        assert client.__name__ == "PromotionGateClient"
        assert agent_eval_kit.GateClientError.__name__ == "GateClientError"
        assert agent_eval_kit.LocalModelJudge.__name__ == "LocalModelJudge"
        print("OK")
        """,
        blocked=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "OK" in result.stdout


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    """A module `__getattr__` that swallows unknown names hides typos; this one must not."""
    import agent_eval_kit

    with pytest.raises(AttributeError, match="no_such_symbol"):
        _ = agent_eval_kit.no_such_symbol  # type: ignore[attr-defined]


def test_the_two_declared_versions_agree() -> None:
    """`__version__` and the pyproject version must never disagree in a shipped release."""
    import tomllib
    from pathlib import Path

    import agent_eval_kit

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert agent_eval_kit.__version__ == declared
