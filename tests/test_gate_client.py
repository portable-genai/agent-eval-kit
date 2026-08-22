"""Contract test for the promotion-gate client.

Guards the client contract: a structured ``target``, top-level ``dataset_id`` equal to
``target.dataset_id``, selection by the registered bundle with no bare metric names, response
parsing from ``results[]``, a POST promotion gate, and injectable auth headers.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent_eval_kit.gate_client import GateClientError, PromotionGateClient

_BASE = "http://localhost:8084"


def _client(**kwargs: object) -> PromotionGateClient:
    return PromotionGateClient(_BASE, bundle="example-bundle", model="gemini-3.5-flash", **kwargs)


_RESULTS_BODY = {
    "n_examples": 12,
    "target": {"model": "gemini-3.5-flash", "prompt_version": "v1", "dataset_id": "golden_cases"},
    "results": [
        {"metric": "sow_groundedness", "score": 0.91, "threshold": 0.80, "passed": True},
        {"metric": "risk_band_accuracy", "score": 0.70, "threshold": 0.85, "passed": False},
        {"metric": "pii_safety", "score": 1.0, "threshold": 0.99, "passed": True},
    ],
    "passed": False,
    "run_id": "run-17",
    "dataset_version": "2026-07-30",
    "dataset_digest": "sha256:abc",
    "evaluator": "gemini-enterprise-agent-evaluation",
    "schema_version": "eval-run/v1",
    "trace_id": "0af7651916cd43dd8448eb211c80319c",
    "correlation_id": "release-42",
    "artifact_refs": ["gs://evidence/run-17.json"],
    "attested": True,
}

_GATE_BODY = {
    "passed": True,
    "eval_report": {
        **_RESULTS_BODY,
        "results": [{"metric": "groundedness", "score": 0.91, "threshold": 0.80, "passed": True}],
        "passed": True,
    },
    "redteam_report": {
        "passed": True,
        "results": [
            {
                "category": "prompt_injection",
                "probe_id": "rt-1",
                "blocked": True,
                "passed": True,
            }
        ],
    },
    "model_card_ref": "gemini-3.5-flash@v1",
    "mrm_evidence_ref": "mrm/gemini-3.5-flash@v1",
}


def _sent(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


@respx.mock
def test_evaluate_sends_structured_target_and_parses_results():
    route = respx.post(f"{_BASE}/v1/evaluations").mock(
        return_value=httpx.Response(200, json=_RESULTS_BODY)
    )
    report = _client().evaluate("eval/datasets/golden_cases.jsonl")

    body = _sent(route.calls.last.request)
    assert body["target"]["model"] == "gemini-3.5-flash"
    assert body["target"]["dataset_id"] == "golden_cases"
    # Top-level dataset_id equals target.dataset_id (the gate service 422s on divergence).
    assert body["dataset_id"] == body["target"]["dataset_id"] == "golden_cases"
    # Selection by registered bundle; NO bare metric names on the wire.
    assert body["bundle"] == "example-bundle"
    assert "metrics" not in body
    # Parsed from results[] (the real key), thresholds passed through.
    assert report.passed is False
    metrics = {r.metric: (r.threshold, r.passed) for r in report.results}
    assert metrics["risk_band_accuracy"] == (0.85, False)
    assert metrics["pii_safety"] == (0.99, True)
    # n_examples is passed through from the service body (0 when absent).
    assert report.n_examples == 12
    assert report.run_id == "run-17"
    assert report.schema_version == "eval-run/v1"
    assert report.artifact_refs == ("gs://evidence/run-17.json",)
    assert report.attested is True


@respx.mock
def test_gate_posts_and_returns_bool():
    route = respx.post(f"{_BASE}/v1/gate").mock(return_value=httpx.Response(200, json=_GATE_BODY))
    assert _client().gate("eval/datasets/golden_cases.jsonl") is True
    body = _sent(route.calls.last.request)
    assert body["bundle"] == "example-bundle"
    assert body["dataset_id"] == body["target"]["dataset_id"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"results": [], "n_examples": 1},
        {"results": [{"metric": "m", "score": 1.0, "threshold": 0.5, "passed": True}]},
        {
            "results": [{"metric": "m", "score": 1.0, "threshold": 0.5, "passed": "yes"}],
            "n_examples": 1,
        },
        {
            "results": [{"metric": "m", "score": 0.1, "threshold": 0.5, "passed": True}],
            "n_examples": 1,
        },
        {
            "results": [{"metric": "m", "score": 1.0, "threshold": 0.5, "passed": True}],
            "n_examples": 1,
            "passed": False,
        },
    ],
)
@respx.mock
def test_evaluate_rejects_missing_or_contradictory_evidence(body):
    respx.post(f"{_BASE}/v1/evaluations").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(GateClientError):
        _client().evaluate("eval/datasets/golden_cases.jsonl")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"passed": "true"},
        {"passed": 1},
        {"passed": True},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "attested": False}},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "run_id": ""}},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "run_id": None}},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "dataset_digest": None}},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "evaluator": None}},
        {**_GATE_BODY, "eval_report": {**_GATE_BODY["eval_report"], "artifact_refs": [""]}},
        {**_GATE_BODY, "mrm_evidence_ref": ""},
        {**_GATE_BODY, "passed": False},
    ],
)
@respx.mock
def test_gate_rejects_non_boolean_verdict(body):
    respx.post(f"{_BASE}/v1/gate").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(GateClientError):
        _client().gate("eval/datasets/golden_cases.jsonl")


@respx.mock
def test_non_2xx_raises():
    respx.post(f"{_BASE}/v1/evaluations").mock(return_value=httpx.Response(422, text="bad bundle"))
    with pytest.raises(GateClientError):
        _client().evaluate("eval/datasets/golden_cases.jsonl")


@respx.mock
def test_transport_error_raises():
    respx.post(f"{_BASE}/v1/gate").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GateClientError, match="failed"):
        _client().gate("eval/datasets/golden_cases.jsonl")


@respx.mock
def test_auth_headers_are_injected_from_a_mapping():
    route = respx.post(f"{_BASE}/v1/gate").mock(return_value=httpx.Response(200, json=_GATE_BODY))
    _client(auth_headers={"Authorization": "Bearer svc-token"}).gate("d.jsonl")
    assert route.calls.last.request.headers["Authorization"] == "Bearer svc-token"


@respx.mock
def test_auth_headers_can_be_a_callable_resolved_per_call():
    route = respx.post(f"{_BASE}/v1/gate").mock(return_value=httpx.Response(200, json=_GATE_BODY))
    _client(auth_headers=lambda: {"Authorization": "Bearer lazy"}).gate("d.jsonl")
    assert route.calls.last.request.headers["Authorization"] == "Bearer lazy"


def test_plaintext_non_loopback_base_url_is_rejected():
    with pytest.raises(ValueError, match="https://"):
        PromotionGateClient("http://gate.example", bundle="b", model="m")


def test_https_and_loopback_base_urls_are_accepted():
    assert PromotionGateClient("https://gate.example/", bundle="b", model="m")
    assert PromotionGateClient("http://127.0.0.1:8084", bundle="b", model="m")
