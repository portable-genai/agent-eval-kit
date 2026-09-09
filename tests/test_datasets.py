"""Golden-set loading, and the three shapes that make a gate report a number over nothing."""

from __future__ import annotations

import io
import json

import pytest

from agent_eval_kit.datasets import (
    DatasetError,
    dataset_digest,
    fraction,
    load_jsonl,
    load_yaml_documents,
    warn_unmeasured,
)


def _write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "# a comment a reviewer can read\n\n"
        + "\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def test_comments_and_blank_lines_are_skipped_so_a_corpus_can_explain_itself(tmp_path):
    path = _write(tmp_path, "g.jsonl", [{"id": "a"}, "# why this case exists", {"id": "b"}])
    assert [row["id"] for row in load_jsonl(path)] == ["a", "b"]


def test_a_case_kind_no_metric_scores_is_refused(tmp_path):
    """The quiet half of the E4 failure: the row still counts toward n_examples."""
    path = _write(tmp_path, "g.jsonl", [{"kind": "routing"}, {"kind": "rooting"}])
    with pytest.raises(DatasetError, match="scored by no metric"):
        load_jsonl(path, kinds={"routing", "identity"})


def test_a_recognised_kind_set_loads(tmp_path):
    path = _write(tmp_path, "g.jsonl", [{"kind": "routing"}])
    assert len(load_jsonl(path, kinds={"routing"})) == 1


def test_a_required_field_is_named_rather_than_raising_a_key_error_later(tmp_path):
    path = _write(tmp_path, "g.jsonl", [{"id": "a", "expected": "x"}, {"id": "b"}])
    with pytest.raises(DatasetError, match="'expected' is required"):
        load_jsonl(path, required=("id", "expected"))


def test_an_empty_dataset_is_refused(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="absent measurement is not a pass"):
        load_jsonl(path)


def test_a_missing_dataset_is_refused(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_a_malformed_line_names_the_line_number(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=":2:"):
        load_jsonl(path)


def test_an_empty_selection_scores_zero_and_never_one():
    """The metric that measured nothing must not become the strongest number in the report."""
    assert fraction([]) == 0.0
    assert fraction([True, True]) == 1.0
    assert fraction([True, False]) == 0.5


def test_the_unmeasured_metric_names_itself_so_its_zero_explains_itself():
    stream = io.StringIO()
    unmeasured = warn_unmeasured(
        [{"kind": "config"}],
        {"journey_integrity": "config", "routing_correctness": "routing"},
        stream=stream,
    )
    assert unmeasured == ["routing_correctness"]
    assert "routing_correctness evaluated nothing" in stream.getvalue()


def test_the_digest_identifies_the_data_and_ignores_the_prose(tmp_path):
    """A reviewer improving an explanation must not read as a corpus change."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('# first note\n{"id": "x"}\n', encoding="utf-8")
    b.write_text('# a much better note\n\n{"id": "x"}\n', encoding="utf-8")
    assert dataset_digest(a) == dataset_digest(b)

    c = tmp_path / "c.jsonl"
    c.write_text('{"id": "y"}\n', encoding="utf-8")
    assert dataset_digest(a) != dataset_digest(c)


def test_yaml_documents_travel_with_their_own_path(tmp_path):
    root = tmp_path / "scenarios"
    (root / "sg").mkdir(parents=True)
    (root / "sg" / "one.yaml").write_text("id: sg-1\nmode: agent_assist\n", encoding="utf-8")
    (root / "sg" / "two.yaml").write_text("id: sg-2\nmode: self_service\n", encoding="utf-8")
    loaded = load_yaml_documents(root)
    assert [path.name for path, _ in loaded] == ["one.yaml", "two.yaml"]
    assert loaded[0][1]["id"] == "sg-1"


def test_an_empty_scenario_tree_is_refused(tmp_path):
    (tmp_path / "scenarios").mkdir()
    with pytest.raises(DatasetError, match="nothing would be scored"):
        load_yaml_documents(tmp_path / "scenarios")
