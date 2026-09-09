"""Replay a real model's committed words offline, and refuse a recording batch on one leak."""

from __future__ import annotations

import json

import pytest

from agent_eval_kit.recording import Recorded, RecordingRefused, write_recordings
from agent_eval_kit.replay import ReplayAdapter, ReplayError, load_recordings, recording_key


def _never_leaks(_text: str) -> bool:
    return False


def _row(key: str, response: object = None) -> Recorded:
    return Recorded(
        key=key,
        case_id="case-1",
        model="a-model",
        response=response if response is not None else {"text": "A grounded reply."},
    )


# --------------------------------------------------------------------------- #
# The key
# --------------------------------------------------------------------------- #
def test_the_key_covers_the_model_the_prompt_and_what_the_model_was_given():
    base = recording_key("m", "p", ["a"])
    assert recording_key("other", "p", ["a"]) != base
    assert recording_key("m", "other", ["a"]) != base
    assert recording_key("m", "p", ["b"]) != base


def test_the_order_evidence_arrived_in_is_not_part_of_the_input():
    assert recording_key("m", "p", ["b", "a"]) == recording_key("m", "p", ["a", "b"])


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #
def test_a_missing_recording_raises_and_never_falls_back(tmp_path):
    fixture = tmp_path / "replay.jsonl"
    fixture.write_text(json.dumps({"key": "other", "response": {"text": "hi"}}) + "\n", "utf-8")
    ReplayAdapter.reset_misses()
    adapter = ReplayAdapter("a-model", fixture)
    with pytest.raises(ReplayError, match="Re-record rather than falling back"):
        adapter.replay("a prompt", ["p1"])


def test_every_miss_is_recorded_so_a_swallowed_raise_still_fails_the_run(tmp_path):
    """A kernel that turns generation failure into silence would otherwise hide a stale file.

    Silence is a scoreable product state, so it passes wherever silence was the expected
    answer. The metrics alone therefore cannot tell a stale recording from a model that
    declined, which is why the miss list exists next to them.
    """
    fixture = tmp_path / "replay.jsonl"
    fixture.write_text(json.dumps({"key": "other", "response": None}) + "\n", "utf-8")
    ReplayAdapter.reset_misses()
    adapter = ReplayAdapter("a-model", fixture)
    with pytest.raises(ReplayError):
        adapter.replay("a prompt")
    with pytest.raises(ReplayError, match="scored something other than the model"):
        ReplayAdapter.assert_no_misses()


def test_a_clean_run_has_no_misses(tmp_path):
    fixture = tmp_path / "replay.jsonl"
    key = recording_key("a-model", "a prompt", ["p1"])
    fixture.write_text(json.dumps({"key": key, "response": {"text": "grounded"}}) + "\n", "utf-8")
    ReplayAdapter.reset_misses()
    adapter = ReplayAdapter("a-model", fixture)
    assert adapter.replay("a prompt", ["p1"]) == {"text": "grounded"}
    ReplayAdapter.assert_no_misses()


def test_a_recorded_null_is_a_real_answer_and_is_replayed_as_silence(tmp_path):
    """The model declined. Raising on it would hide a case worth scoring."""
    fixture = tmp_path / "replay.jsonl"
    key = recording_key("a-model", "a prompt", [])
    fixture.write_text(json.dumps({"key": key, "response": None}) + "\n", "utf-8")
    ReplayAdapter.reset_misses()
    assert ReplayAdapter("a-model", fixture).replay("a prompt") is None
    ReplayAdapter.assert_no_misses()


def test_an_absent_fixture_says_how_to_record_it(tmp_path):
    with pytest.raises(ReplayError, match="measurement of the validator"):
        load_recordings(tmp_path / "nothing.jsonl")


def test_a_fixture_of_only_comments_is_refused(tmp_path):
    fixture = tmp_path / "replay.jsonl"
    fixture.write_text("# recorded once, by hand\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="no recordings"):
        load_recordings(fixture)


def test_a_row_with_no_key_cannot_be_replayed(tmp_path):
    fixture = tmp_path / "replay.jsonl"
    fixture.write_text(json.dumps({"response": {"text": "x"}}) + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="must carry a 'key'"):
        load_recordings(fixture)


# --------------------------------------------------------------------------- #
# The recording
# --------------------------------------------------------------------------- #
def test_one_leak_refuses_the_whole_batch_and_writes_nothing(tmp_path):
    """Not a filter. A filter writes a scrubbed lie and loses the case that leaked."""
    out = tmp_path / "replay.jsonl"
    rows = [_row("k1"), _row("k2", {"text": "the NRIC is S1234567D"}), _row("k3")]
    with pytest.raises(RecordingRefused, match="Nothing has been written"):
        write_recordings(out, rows, leaks=lambda text: "S1234567D" in text)
    assert not out.exists(), "a partial write is how a scrubbed file ends up half scrubbed"


def test_the_planted_literal_is_a_second_oracle_that_fires_when_the_pattern_row_is_broken(
    tmp_path,
):
    out = tmp_path / "replay.jsonl"
    rows = [_row("k1", {"text": "your reference is ACME-PLANTED-42"})]
    with pytest.raises(RecordingRefused, match="planted identifier"):
        write_recordings(out, rows, leaks=_never_leaks, planted=["ACME-PLANTED-42"])
    assert not out.exists()


def test_an_over_long_reply_is_refused_because_the_validator_would_reject_it(tmp_path):
    out = tmp_path / "replay.jsonl"
    with pytest.raises(RecordingRefused, match="over the 10"):
        write_recordings(out, [_row("k1", {"text": "x" * 11})], leaks=_never_leaks, max_chars=10)


def test_an_empty_batch_is_refused_rather_than_written(tmp_path):
    """An empty file would make every replayed case a miss, hiding that capture never ran."""
    with pytest.raises(RecordingRefused, match="nothing was captured"):
        write_recordings(tmp_path / "replay.jsonl", [], leaks=_never_leaks)


def test_a_clean_batch_is_written_with_its_header_and_replays_back(tmp_path):
    out = tmp_path / "replay.jsonl"
    key = recording_key("a-model", "a prompt", ["p1"])
    written = write_recordings(
        out,
        [Recorded(key=key, case_id="c1", model="a-model", response={"text": "grounded"})],
        leaks=_never_leaks,
        planted=["S1234567D"],
    )
    assert written == 1
    assert out.read_text(encoding="utf-8").startswith("# Recorded managed-model replies")
    ReplayAdapter.reset_misses()
    assert ReplayAdapter("a-model", out).replay("a prompt", ["p1"]) == {"text": "grounded"}
