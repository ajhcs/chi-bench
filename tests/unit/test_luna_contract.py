"""Offline Luna contracts. Prepared without running repository code."""

from dataclasses import dataclass
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from chi_bench.experiment.agents.luna_contract import (
    LUNA_ENDPOINT, LUNA_MODEL, LunaBudgetExceeded, LunaContract, LunaContractError, LunaLedger,
)


def luna_env():
    return {
        "OPENAI_AGENTS_MODEL": LUNA_MODEL,
        "OPENAI_BASE_URL": LUNA_ENDPOINT,
        "OPENAI_AGENTS_API_MODE": "responses",
        "OPENAI_AGENTS_REASONING_EFFORT": "medium",
        "OPENAI_AGENTS_REASONING_MODE": "standard",
        "OPENAI_AGENTS_MAX_RETRIES": "0",
        "OPENAI_AGENTS_MAX_OUTPUT_TOKENS": "8",
        "OPENAI_AGENTS_MAX_TOTAL_OUTPUT_TOKENS": "10",
        "OPENAI_AGENTS_MAX_MODEL_CALLS": "2",
        "OPENAI_AGENTS_MAX_TOTAL_TOKENS": "100",
    }


def response(inp=2, out=3):
    return SimpleNamespace(
        response_id="resp_synthetic", request_id="req_synthetic", output=[],
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                              input_tokens_details=SimpleNamespace(cached_tokens=1),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=2)),
    )


@pytest.mark.parametrize("effort", ["medium", "max"])
@pytest.mark.parametrize("mode", ["standard", "pro"])
def test_exact_requested_effort_and_mode(effort, mode):
    env = luna_env()
    env.update(OPENAI_AGENTS_REASONING_EFFORT=effort, OPENAI_AGENTS_REASONING_MODE=mode)
    contract = LunaContract.from_env(env)
    assert (contract.effort, contract.mode) == (effort, mode)


@pytest.mark.parametrize("key,value", [
    ("OPENAI_AGENTS_MODEL", "gpt-5.6-luna"),
    ("OPENAI_AGENTS_MODEL", "openai/gpt-6-luna"),
    ("OPENAI_BASE_URL", "https://example.invalid/v1"),
    ("OPENAI_AGENTS_API_MODE", "chat_completions"),
    ("OPENAI_AGENTS_REASONING_EFFORT", "xhigh"),
    ("OPENAI_AGENTS_REASONING_MODE", ""),
    ("OPENAI_AGENTS_MAX_RETRIES", "1"),
    ("OPENAI_AGENTS_MAX_OUTPUT_TOKENS", "0"),
    ("OPENAI_AGENTS_MAX_TOTAL_OUTPUT_TOKENS", "-1"),
    ("OPENAI_AGENTS_MAX_MODEL_CALLS", "1.5"),
    ("OPENAI_AGENTS_MAX_TOTAL_TOKENS", ""),
])
def test_rejects_implicit_or_substituted_contract(key, value):
    env = luna_env()
    env[key] = value
    with pytest.raises(LunaContractError):
        LunaContract.from_env(env)


def test_output_reservation_clamps_next_call_and_stops_at_limit():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    first, limit = ledger.admit()
    assert limit == 8
    ledger.observe(first, response(out=8))
    second, limit = ledger.admit()
    assert limit == 2
    ledger.observe(second, response(out=2))
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()
    assert ledger.output_tokens == 10
    assert ledger.snapshot()["cost_usd"] is None


def test_pending_or_missing_usage_cannot_be_retried_as_free():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    record, _ = ledger.admit()
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()
    with pytest.raises(LunaBudgetExceeded):
        ledger.observe(record, SimpleNamespace())
    assert ledger.snapshot()["usage_status"] == "unknown"
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


def test_observed_total_guard_preserves_usage_and_blocks_further_calls():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    record, _ = ledger.admit()
    with pytest.raises(LunaBudgetExceeded, match="observed_total"):
        ledger.observe(record, response(inp=100, out=1))
    assert ledger.input_tokens == 100
    assert ledger.complete
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort,mode", [("medium", "standard"), ("max", "pro")])
async def test_sdk_boundary_receives_limits_reasoning_and_no_sampling(monkeypatch, effort, mode):
    from chi_bench.experiment.agents import openai_agents_runner as runner

    env = luna_env()
    env.update(OPENAI_AGENTS_REASONING_EFFORT=effort, OPENAI_AGENTS_REASONING_MODE=mode)
    ledger = LunaLedger(LunaContract.from_env(env))
    calls, snapshots = [], []
    interface = ModuleType("agents.models.interface")
    interface.Model = object
    responses = ModuleType("agents.models.openai_responses")

    class Delegate:
        def __init__(self, model, openai_client):
            assert model == LUNA_MODEL

        async def get_response(self, **kwargs):
            calls.append(kwargs["model_settings"])
            return response(out=8 if len(calls) == 1 else 2)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"base_url": LUNA_ENDPOINT, "max_retries": 0}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    responses.OpenAIResponsesModel = Delegate
    openai = ModuleType("openai")
    openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "agents.models.interface", interface)
    monkeypatch.setitem(sys.modules, "agents.models.openai_responses", responses)
    monkeypatch.setitem(sys.modules, "openai", openai)

    @dataclass
    class Settings:
        max_tokens: int = 999
        temperature: float | None = 1.0
        top_p: float | None = 1.0
        reasoning: dict | None = None

    async with runner._agent_model_context(
        LUNA_MODEL, "responses", ledger=ledger,
        persist=lambda: snapshots.append(json.loads(json.dumps(ledger.snapshot()))),
    ) as model:
        await model.get_response(model_settings=Settings())
        await model.get_response(model_settings=Settings())
        with pytest.raises(LunaBudgetExceeded):
            await model.get_response(model_settings=Settings())
    assert [s.max_tokens for s in calls] == [8, 2]
    assert all(s.reasoning == {"effort": effort, "mode": mode} for s in calls)
    assert all(s.temperature is None and s.top_p is None for s in calls)
    assert snapshots[0]["requests"][0]["status"] == "pending"
    assert snapshots[-1]["usage_status"] == "complete"


@pytest.mark.asyncio
async def test_generic_runner_exception_is_unknown_usage_not_zero(monkeypatch, tmp_path):
    from tests.unit.test_openai_agents_runner_config import _install_fake_agents_sdk
    from chi_bench.experiment.agents import openai_agents_runner as runner

    _install_fake_agents_sdk(monkeypatch)
    monkeypatch.setenv("OPENAI_AGENTS_MODEL", "gpt-5.6-sol")

    async def fail(**kwargs):
        raise RuntimeError("synthetic secret-bearing error must not be serialized")

    monkeypatch.setattr(sys.modules["agents"].Runner, "run", fail)
    with pytest.raises(SystemExit):
        await runner.run_agent("synthetic", "http://example.invalid/mcp", tmp_path)
    result = json.loads((tmp_path / "run_result.json").read_text())
    assert result["input_tokens"] is None
    assert result["cost_usd"] is None
    assert result["usage_status"] == "unknown"
    assert result["error"] == "RuntimeError"
    assert "secret-bearing" not in json.dumps(result)


def test_completed_usage_survives_later_unknown_request():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    first, _ = ledger.admit()
    ledger.observe(first, response(inp=5, out=2))
    second, _ = ledger.admit()
    second.update(status="error_usage_unknown", error_type="TimeoutError")
    snapshot = ledger.snapshot()
    assert snapshot["usage_status"] == "unknown"
    assert snapshot["observed_input_tokens"] == 5
    assert snapshot["observed_output_tokens"] == 2
    assert snapshot["requests"][0]["request_id"] == "req_synthetic"
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


def test_missing_cache_or_reasoning_detail_remains_unknown():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    record, _ = ledger.admit()
    reply = response()
    reply.usage.input_tokens_details = None
    reply.usage.output_tokens_details = None
    ledger.observe(record, reply)
    assert ledger.snapshot()["observed_cached_tokens"] is None
    assert ledger.snapshot()["observed_reasoning_tokens"] is None
