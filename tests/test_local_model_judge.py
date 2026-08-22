"""Contract test for the OPT-IN model judge, run entirely offline.

Nothing here talks to a real server, and no test in this package depends on one being up: the
whole point of the judge harness is that the gate runs with no model. ``respx`` mocks the
chat-completions endpoint so the wire contract, and every fail-closed refusal, is pinned without
a process to start. A live server is exercised by running the judge deliberately, not by a test
that quietly skips itself into green.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent_eval_kit.judge import (
    JUDGE_BACKEND_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    LOCAL_MODEL_BACKEND,
    JudgeRequest,
    JudgeSelection,
    JudgeUnavailableError,
    NarrativeCriterion,
    build_judge,
)
from agent_eval_kit.local_model_judge import LocalModelJudge

_BASE = "http://127.0.0.1:8001"
_MODEL = "mlx-community/example-model-4bit"
_CRITERIA = (
    NarrativeCriterion(name="explains_the_finding", must_cover=("the nominee shareholder",)),
    NarrativeCriterion(
        name="stays_within_evidence", must_not_say=("guaranteed clean",), weight=2.0
    ),
)
_REQUEST = JudgeRequest(candidate="A nominee shareholder holds the stake.", criteria=_CRITERIA)


def _judge(**kwargs: object) -> LocalModelJudge:
    return LocalModelJudge(_BASE, model=_MODEL, **kwargs)  # type: ignore[arg-type]


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


_GOOD_CONTENT = (
    '{"scores": [{"criterion": "explains_the_finding", "score": 0.9, "detail": "covered"}, '
    '{"criterion": "stays_within_evidence", "score": 0.6, "detail": "hedged"}]}'
)


@respx.mock
def test_the_wire_contract_has_no_v1_prefix_and_asks_for_deterministic_decoding() -> None:
    """The local MLX server speaks OpenAI-compatible paths WITHOUT the ``/v1`` prefix."""
    route = respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(_GOOD_CONTENT))
    _judge().grade(_REQUEST)
    assert route.called
    body = json.loads(route.calls.last.request.read())
    assert body["model"] == _MODEL
    assert body["temperature"] == 0.0
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    # The criteria travel in the prompt, so the model grades the SAME data the offline judge does.
    assert "explains_the_finding" in body["messages"][1]["content"]
    assert "guaranteed clean" in body["messages"][1]["content"]


@respx.mock
def test_a_well_formed_reply_becomes_a_weighted_verdict() -> None:
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(_GOOD_CONTENT))
    verdict = _judge().grade(_REQUEST)
    assert verdict.graded_by == f"local-model:{_MODEL}"
    assert verdict.has_evidence is True
    # The criteria weights come from the REQUEST, not from whatever the model claims.
    assert verdict.score == pytest.approx((0.9 * 1.0 + 0.6 * 2.0) / 3.0)


@respx.mock
def test_a_fenced_and_thinking_wrapped_reply_is_still_parsed() -> None:
    wrapped = f"<think>weighing {{the}} options</think>\n```json\n{_GOOD_CONTENT}\n```"
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(wrapped))
    assert _judge().grade(_REQUEST).has_evidence is True


@respx.mock
def test_a_reply_with_no_scores_is_refused_rather_than_read_as_a_pass() -> None:
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply('{"scores": []}'))
    with pytest.raises(JudgeUnavailableError, match="empty verdict is not a pass"):
        _judge().grade(_REQUEST)


@respx.mock
def test_a_partial_verdict_is_refused() -> None:
    """Half the criteria graded, scored against a floor that assumes all of them, reads high."""
    content = '{"scores": [{"criterion": "explains_the_finding", "score": 1.0}]}'
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(content))
    with pytest.raises(JudgeUnavailableError, match="did not grade stays_within_evidence"):
        _judge().grade(_REQUEST)


@respx.mock
def test_a_criterion_the_judge_invented_is_refused() -> None:
    content = '{"scores": [{"criterion": "vibes", "score": 1.0}]}'
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(content))
    with pytest.raises(JudgeUnavailableError, match="not one of the criteria"):
        _judge().grade(_REQUEST)


@respx.mock
def test_a_duplicated_criterion_is_refused() -> None:
    content = (
        '{"scores": [{"criterion": "explains_the_finding", "score": 0.1}, '
        '{"criterion": "explains_the_finding", "score": 1.0}]}'
    )
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(content))
    with pytest.raises(JudgeUnavailableError, match="more than once"):
        _judge().grade(_REQUEST)


@respx.mock
def test_a_score_outside_the_unit_interval_is_refused() -> None:
    content = (
        '{"scores": [{"criterion": "explains_the_finding", "score": 9}, '
        '{"criterion": "stays_within_evidence", "score": 1.0}]}'
    )
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply(content))
    with pytest.raises(JudgeUnavailableError, match=r"outside \[0, 1\]"):
        _judge().grade(_REQUEST)


@respx.mock
def test_prose_instead_of_json_is_refused() -> None:
    respx.post(f"{_BASE}/chat/completions").mock(return_value=_reply("Looks good to me!"))
    with pytest.raises(JudgeUnavailableError, match="no JSON object"):
        _judge().grade(_REQUEST)


@respx.mock
def test_a_server_error_is_refused_rather_than_scored() -> None:
    respx.post(f"{_BASE}/chat/completions").mock(return_value=httpx.Response(503, text="busy"))
    with pytest.raises(JudgeUnavailableError, match="returned 503"):
        _judge().grade(_REQUEST)


@respx.mock
def test_an_unreachable_server_raises_and_never_returns_a_default_score() -> None:
    respx.post(f"{_BASE}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(JudgeUnavailableError, match="failed"):
        _judge().grade(_REQUEST)


@respx.mock
def test_probe_reports_reachability_without_inventing_a_verdict() -> None:
    respx.get(f"{_BASE}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    assert _judge().probe() is True
    respx.get(f"{_BASE}/models").mock(side_effect=httpx.ConnectError("refused"))
    assert _judge().probe() is False


def test_a_plaintext_remote_judge_url_is_refused() -> None:
    """The narrative and its criteria travel on this connection, so loopback or https only."""
    with pytest.raises(ValueError, match="https://"):
        LocalModelJudge("http://judge.example.invalid", model=_MODEL)


def test_an_unnamed_model_is_refused() -> None:
    with pytest.raises(ValueError, match="model id"):
        LocalModelJudge(_BASE, model="  ")


def test_the_selection_builds_this_adapter_when_it_is_named() -> None:
    judge = build_judge(
        JudgeSelection.from_env(
            {
                JUDGE_BACKEND_ENV: LOCAL_MODEL_BACKEND,
                JUDGE_BASE_URL_ENV: _BASE,
                JUDGE_MODEL_ENV: _MODEL,
            }
        )
    )
    assert isinstance(judge, LocalModelJudge)
    assert judge.name == f"local-model:{_MODEL}"


@respx.mock
def test_a_custom_path_reaches_a_server_that_does_use_the_v1_prefix() -> None:
    route = respx.post(f"{_BASE}/v1/chat/completions").mock(return_value=_reply(_GOOD_CONTENT))
    _judge(path="/v1/chat/completions").grade(_REQUEST)
    assert route.called
