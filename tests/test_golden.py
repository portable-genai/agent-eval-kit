"""Render-and-check: inputs from the shipped demo data, expectations hand-written, staleness red."""

from __future__ import annotations

import io

import pytest

from agent_eval_kit.golden import GoldenRenderError, check, render_main, write


def _render() -> str:
    return '# GENERATED\n{"id": "client-1", "expected_band": "amber"}\n'


def test_write_then_check_is_green(tmp_path):
    out = tmp_path / "eval" / "datasets" / "golden_clients.jsonl"
    assert write(out, _render) == 1
    assert check(out, _render, stream=io.StringIO()) is True


def test_a_stale_committed_file_fails_and_prints_the_diff(tmp_path):
    out = tmp_path / "golden.jsonl"
    write(out, _render)
    stream = io.StringIO()

    def moved() -> str:
        return '# GENERATED\n{"id": "client-1", "expected_band": "red"}\n'

    assert check(out, moved, stream=stream) is False
    printed = stream.getvalue()
    assert "expected_band" in printed
    assert "is stale" in printed
    assert "rendered from its sources" in printed


def test_a_missing_committed_file_says_to_render_it(tmp_path):
    stream = io.StringIO()
    assert check(tmp_path / "absent.jsonl", _render, stream=stream) is False
    assert "run the renderer" in stream.getvalue()


def test_a_renderer_that_produces_nothing_is_refused(tmp_path):
    """Writing it would empty the golden set and every metric would average no cases."""
    with pytest.raises(GoldenRenderError, match="produced nothing"):
        write(tmp_path / "golden.jsonl", lambda: "   \n")


def test_the_cli_writes_by_default_and_verifies_with_check(tmp_path, capsys):
    out = tmp_path / "golden.jsonl"
    assert render_main(output=out, render=_render, argv=[]) == 0
    assert "wrote 1 case(s)" in capsys.readouterr().out
    assert render_main(output=out, render=_render, argv=["--check"]) == 0

    out.write_text('# GENERATED\n{"id": "client-1", "expected_band": "green"}\n', "utf-8")
    assert render_main(output=out, render=_render, argv=["--check"]) == 1


def test_a_missing_trailing_newline_is_not_a_diff(tmp_path):
    """A renderer that forgets the final newline must not make every check report staleness."""
    out = tmp_path / "golden.jsonl"
    write(out, lambda: '{"id": "a"}')
    assert check(out, lambda: '{"id": "a"}\n', stream=io.StringIO()) is True
