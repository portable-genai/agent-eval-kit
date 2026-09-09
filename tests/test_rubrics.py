"""Thresholds as reviewed data, and the two symmetrical refusals that keep them honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_kit.rubrics import RubricError, load_rubrics

_YAML = """\
# Scored by: eval/run_eval.py. 0.99 rather than 1.0 because one borderline near-miss in a
# corpus this size is a tuning signal, not a release blocker.
metric: citation_grounding
description: Every asserted figure resolves to a retrieved passage.
threshold: 0.99
companion_metrics:
  pii_safety:
    threshold: 0.99
    description: No raw identifier survives into any emitted record.
"""


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def test_a_rubric_carries_the_headline_metric_and_its_companions(tmp_path):
    rubrics = load_rubrics(_tree(tmp_path / "rubrics", {"grounding.yaml": _YAML}))
    assert rubrics.thresholds() == {"citation_grounding": 0.99, "pii_safety": 0.99}
    assert "resolves to a retrieved passage" in rubrics["citation_grounding"].description


def test_toml_and_json_rubrics_need_no_third_party_parser(tmp_path):
    root = _tree(
        tmp_path / "rubrics",
        {
            "a.toml": 'metric = "grade_accuracy"\nthreshold = 1.0\n',
            "b.json": json.dumps({"metric": "routing_accuracy", "threshold": 1.0}),
        },
    )
    assert load_rubrics(root).thresholds() == {"grade_accuracy": 1.0, "routing_accuracy": 1.0}


def test_a_metric_with_no_reviewed_bar_fails_the_build(tmp_path):
    rubrics = load_rubrics(_tree(tmp_path / "rubrics", {"g.yaml": _YAML}))
    with pytest.raises(RubricError, match="invented at the call site"):
        rubrics.assert_covers(["citation_grounding", "pii_safety", "typology_recall"])


def test_a_bar_that_names_no_metric_fails_the_build_too(tmp_path):
    """The half nobody writes by hand, and the one that rots toward looking well governed.

    A rubric for a metric that was renamed or deleted keeps reading as governance, satisfies a
    reviewer, and gates nothing at all.
    """
    rubrics = load_rubrics(_tree(tmp_path / "rubrics", {"g.yaml": _YAML}))
    with pytest.raises(RubricError, match="reads as governance"):
        rubrics.assert_covers(["citation_grounding"])


def test_both_directions_together_are_the_passing_case(tmp_path):
    rubrics = load_rubrics(_tree(tmp_path / "rubrics", {"g.yaml": _YAML}))
    rubrics.assert_covers(["citation_grounding", "pii_safety"])


def test_one_metric_one_bar(tmp_path):
    root = _tree(
        tmp_path / "rubrics",
        {
            "a.yaml": "metric: safety\nthreshold: 0.99\n",
            "b.yaml": "metric: safety\nthreshold: 0.90\n",
        },
    )
    with pytest.raises(RubricError, match="One metric, one bar"):
        load_rubrics(root)


def test_a_missing_rubric_directory_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(RubricError, match="no rubric directory"):
        load_rubrics(tmp_path / "not-here")


def test_a_rubric_naming_no_metric_is_refused(tmp_path):
    root = _tree(tmp_path / "rubrics", {"a.yaml": "description: a note\nthreshold: 0.9\n"})
    with pytest.raises(RubricError, match="names no metric"):
        load_rubrics(root)


def test_a_non_numeric_or_out_of_range_bar_is_refused(tmp_path):
    root = _tree(tmp_path / "rubrics", {"a.yaml": 'metric: safety\nthreshold: "high"\n'})
    with pytest.raises(RubricError, match="non-numeric"):
        load_rubrics(root)
    root = _tree(tmp_path / "rubrics2", {"a.yaml": "metric: safety\nthreshold: 1.5\n"})
    with pytest.raises(RubricError, match="outside"):
        load_rubrics(root)


def test_a_companion_with_no_threshold_is_a_note_pretending_to_be_a_gate(tmp_path):
    body = "metric: a\nthreshold: 1.0\ncompanion_metrics:\n  b:\n    description: only prose\n"
    with pytest.raises(RubricError, match="not a note"):
        load_rubrics(_tree(tmp_path / "rubrics", {"a.yaml": body}))


def test_groups_gate_two_releases_separately(tmp_path):
    root = _tree(
        tmp_path / "rubrics",
        {
            "agent_assist/a.yaml": "metric: next_step_accuracy\nthreshold: 1.0\n",
            "self_service/b.yaml": "metric: gate_precision\nthreshold: 1.0\n",
        },
    )
    rubrics = load_rubrics(root, groups=("agent_assist", "self_service"))
    assert rubrics.group("agent_assist").metrics == ("next_step_accuracy",)
    assert rubrics.group("self_service").metrics == ("gate_precision",)


def test_a_named_group_with_no_directory_is_an_error_not_an_empty_filter(tmp_path):
    """The release it gates would otherwise ship with no bars at all."""
    root = _tree(tmp_path / "rubrics", {"agent_assist/a.yaml": "metric: m\nthreshold: 1.0\n"})
    with pytest.raises(RubricError, match="has no thresholds"):
        load_rubrics(root, groups=("agent_assist", "self_service"))


def test_it_reads_the_real_exemplar_tree_in_this_workspace():
    """Loaded against a tree this kit did not write, when that tree is checked out beside it."""
    root = Path(__file__).resolve().parents[2] / "contact-centre-conversations" / "eval" / "rubrics"
    if not root.is_dir():  # pragma: no cover - the kit is also built standalone
        pytest.skip("the exemplar repository is not checked out beside this one")
    rubrics = load_rubrics(root, groups=("agent_assist", "self_service"))
    assert rubrics.group("self_service")["gate_precision"].threshold == 1.0
    assert rubrics.group("agent_assist")["pii_safety"].threshold == 0.99


# --------------------------------------------------------------------------- #
# Groups are independent, because a repository that gates two families reuses names
# --------------------------------------------------------------------------- #
def test_two_groups_may_carry_the_same_metric_name_with_different_bars(tmp_path):
    """The case that proved the old keying wrong.

    A repository gating a design review and a data-residency scan in one run legitimately has a
    `citation_accuracy` in both: the same words, different questions, and possibly different
    bars. Keyed by name alone, whichever file sorted last won and the other family's rubric
    vanished from the tree while still sitting on disk looking authoritative.
    """
    root = _tree(
        tmp_path / "rubrics",
        {
            "citation_accuracy.yaml": "metric: citation_accuracy\nthreshold: 0.90\n",
            "residency/citation_accuracy.yaml": "metric: citation_accuracy\nthreshold: 0.99\n",
        },
    )
    rubrics = load_rubrics(root)
    assert rubrics.group("")["citation_accuracy"].threshold == 0.90
    assert rubrics.group("residency")["citation_accuracy"].threshold == 0.99
    assert len(rubrics) == 2
    assert rubrics.groups == ("", "residency")


def test_the_same_metric_at_the_same_bar_in_two_groups_is_fine(tmp_path):
    """Equal bars are not a conflict, and each group still holds its own rubric."""
    root = _tree(
        tmp_path / "rubrics",
        {
            "a/safety.yaml": "metric: safety\nthreshold: 0.99\n",
            "b/safety.yaml": "metric: safety\nthreshold: 0.99\n",
        },
    )
    rubrics = load_rubrics(root)
    assert len(rubrics) == 2
    assert rubrics.group("a")["safety"].source.endswith("a/safety.yaml")
    assert rubrics.group("b")["safety"].source.endswith("b/safety.yaml")


def test_one_metric_one_bar_still_holds_inside_a_group(tmp_path):
    """The rule keeps its meaning where it was always aimed: two files, one directory."""
    root = _tree(
        tmp_path / "rubrics",
        {
            "residency/one.yaml": "metric: safety\nthreshold: 0.99\n",
            "residency/two.yaml": "metric: safety\nthreshold: 0.90\n",
        },
    )
    with pytest.raises(RubricError, match="One metric, one bar"):
        load_rubrics(root)


def test_a_group_scoped_assert_covers_does_not_see_the_other_group(tmp_path):
    """Narrowing to a group is what makes `assert_covers` answerable for a two-family repo."""
    root = _tree(
        tmp_path / "rubrics",
        {
            "principle_accuracy.yaml": "metric: principle_accuracy\nthreshold: 0.90\n",
            "residency/detection.yaml": "metric: detection_recall\nthreshold: 0.90\n",
        },
    )
    rubrics = load_rubrics(root)
    rubrics.group("").assert_covers(["principle_accuracy"])
    rubrics.group("residency").assert_covers(["detection_recall"])
    with pytest.raises(RubricError, match="reads as governance"):
        rubrics.assert_covers(["principle_accuracy"])
