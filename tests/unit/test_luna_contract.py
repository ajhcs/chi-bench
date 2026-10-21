"""Prepared offline Luna admission and raw-receipt tests; never live APIs."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from chi_bench.experiment.agents.luna_contract import (
    LUNA_CONTEXT_TOKENS, LUNA_ENDPOINT, LUNA_MODEL, LunaBudgetExceeded,
    LunaContract, LunaContractError, LunaLedger, LunaResponsesBoundary,
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
        "OPENAI_AGENTS_MAX_TOTAL_INPUT_TOKENS": "2100000",
        "OPENAI_AGENTS_MAX_MODEL_CALLS": "2",
        "OPENAI_AGENTS_MAX_TOTAL_TOKENS": "2100010",
    }


def response(inp=2, out=3, **overrides):
    raw = {
        "id": "resp_synthetic", "model": LUNA_MODEL, "status": "completed",
        "reasoning": {"effort": "medium", "mode": "standard"},
        "service_tier": "default",
        "usage": {"input_tokens": inp, "output_tokens": out,
                  "input_tokens_details": {"cached_tokens": 1},
                  "output_tokens_details": {"reasoning_tokens": 2}},
        "output": [],
    }
    raw.update(overrides)
    return SimpleNamespace(model_dump=lambda **kwargs: deepcopy(raw), _request_id="req_synthetic")


@pytest.mark.parametrize("effort", ["medium", "max"])
def test_exact_requested_effort(effort):
    env = luna_env()
    env["OPENAI_AGENTS_REASONING_EFFORT"] = effort
    contract = LunaContract.from_env(env)
    assert (contract.effort, contract.mode) == (effort, "standard")


@pytest.mark.parametrize("key,value", [
    ("OPENAI_AGENTS_MODEL", "gpt-5.6-luna"),
    ("OPENAI_AGENTS_MODEL", "openai/gpt-6-luna"),
    ("OPENAI_BASE_URL", "https://example.invalid/v1"),
    ("OPENAI_AGENTS_API_MODE", "chat_completions"),
    ("OPENAI_AGENTS_REASONING_EFFORT", "xhigh"),
    ("OPENAI_AGENTS_REASONING_MODE", ""),
    ("OPENAI_AGENTS_REASONING_MODE", "pro"),
    ("OPENAI_AGENTS_MAX_RETRIES", "1"),
    ("OPENAI_AGENTS_MAX_OUTPUT_TOKENS", "0"),
    ("OPENAI_AGENTS_MAX_OUTPUT_TOKENS", "128001"),
    ("OPENAI_AGENTS_MAX_TOTAL_OUTPUT_TOKENS", "-1"),
    ("OPENAI_AGENTS_MAX_TOTAL_INPUT_TOKENS", ""),
    ("OPENAI_AGENTS_MAX_MODEL_CALLS", "1.5"),
    ("OPENAI_AGENTS_MAX_TOTAL_TOKENS", ""),
])
def test_rejects_implicit_or_unbounded_contract(key, value):
    env = luna_env()
    env[key] = value
    with pytest.raises(LunaContractError):
        LunaContract.from_env(env)


def test_reservations_are_not_reclaimed_after_small_actual_usage():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    first, limit = ledger.admit()
    assert (first["reserved_input_tokens"], limit) == (LUNA_CONTEXT_TOKENS, 8)
    ledger.observe_raw(first, response(out=3))
    second, limit = ledger.admit()
    assert limit == 2  # Do not reclaim the first eight-token output reservation.
    ledger.observe_raw(second, response(out=2))
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()
    assert ledger.snapshot()["reserved_input_tokens"] == 2 * LUNA_CONTEXT_TOKENS
    assert ledger.snapshot()["reserved_output_tokens"] == 10
    assert ledger.snapshot()["cost_usd"] is None


def test_pending_or_missing_usage_cannot_be_retried_as_free():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    record, _ = ledger.admit()
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()
    with pytest.raises(LunaContractError):
        ledger.observe_raw(record, response(usage=None))
    assert ledger.snapshot()["usage_status"] == "unknown"
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["OPENAI_AGENTS_MAX_TOTAL_INPUT_TOKENS",
                                   "OPENAI_AGENTS_MAX_TOTAL_TOKENS"])
async def test_input_ceiling_admission_precedes_every_transport_call(budget):
    env = luna_env()
    env[budget] = str(LUNA_CONTEXT_TOKENS - 1)
    ledger = LunaLedger(LunaContract.from_env(env))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)

    boundary = LunaResponsesBoundary(SimpleNamespace(create=create), ledger, lambda: None)
    with pytest.raises(LunaBudgetExceeded):
        await boundary.create(model=LUNA_MODEL, reasoning={"effort": "medium"}, stream=False)
    assert calls == [] and ledger.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"model": "another-model"}, {"status": "incomplete"}, {"status": "failed"},
    {"reasoning": {"effort": "xhigh", "mode": "standard"}},
    {"reasoning": {"effort": "medium", "mode": "pro"}},
    {"reasoning": {}}, {"service_tier": "fast"},
])
async def test_raw_provider_mismatch_is_persisted_and_never_accepted(changes):
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    snapshots = []

    async def create(**kwargs):
        return response(**changes)

    boundary = LunaResponsesBoundary(
        SimpleNamespace(create=create), ledger,
        lambda: snapshots.append(deepcopy(ledger.snapshot())),
    )
    with pytest.raises(LunaContractError):
        await boundary.create(model=LUNA_MODEL, reasoning={"effort": "medium"}, stream=False)
    record = snapshots[-1]["requests"][0]
    assert record["provider_contract"] == "invalid"
    assert record["raw_provider_receipt"][next(iter(changes))] == next(iter(changes.values()))
    assert record["request_id"] == "req_synthetic"
    assert ledger.input_tokens == 2
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


@pytest.mark.asyncio
async def test_pending_journal_exists_before_an_ambiguous_transport_failure():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    snapshots = []

    async def create(**kwargs):
        assert snapshots[-1]["requests"][0]["status"] == "pending"
        assert snapshots[-1]["reserved_input_tokens"] == LUNA_CONTEXT_TOKENS
        raise TimeoutError("Do not serialize arbitrary exception bodies")

    boundary = LunaResponsesBoundary(
        SimpleNamespace(create=create), ledger,
        lambda: snapshots.append(deepcopy(ledger.snapshot())),
    )
    with pytest.raises(TimeoutError):
        await boundary.create(model=LUNA_MODEL, reasoning={"effort": "medium"}, stream=False)
    assert snapshots[-1]["usage_status"] == "unknown"
    assert snapshots[-1]["requests"][0]["error_type"] == "TimeoutError"
    assert "arbitrary" not in json.dumps(snapshots)


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["medium", "max"])
async def test_public_sdk_client_derivation_cannot_bypass_raw_receipt_boundary(monkeypatch, effort):
    from chi_bench.experiment.agents import openai_agents_runner as runner

    env = luna_env()
    env["OPENAI_AGENTS_REASONING_EFFORT"] = effort
    ledger = LunaLedger(LunaContract.from_env(env))
    calls, snapshots = [], []
    responses_module = ModuleType("agents.models.openai_responses")

    class Delegate:
        def __init__(self, model, openai_client):
            assert model == LUNA_MODEL
            self.client = openai_client

        async def get_response(self):
            # Mirrors the pinned SDK's retry-disabled public client derivation.
            client = self.client.with_options(max_retries=0)
            return await client.responses.create(
                model=LUNA_MODEL, input=[{"role": "user", "content": "synthetic"}],
                reasoning={"effort": effort}, stream=False,
            )

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"base_url": LUNA_ENDPOINT, "max_retries": 0}
            self.responses = self

        async def create(self, **kwargs):
            calls.append(kwargs)
            return response(out=3 if len(calls) == 1 else 2,
                            reasoning={"effort": effort, "mode": "standard"})

        def with_options(self, **kwargs):
            assert kwargs == {"max_retries": 0}
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    responses_module.OpenAIResponsesModel = Delegate
    openai = ModuleType("openai")
    openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "agents.models.openai_responses", responses_module)
    monkeypatch.setitem(sys.modules, "openai", openai)
    async with runner._agent_model_context(
        LUNA_MODEL, "responses", ledger=ledger,
        persist=lambda: snapshots.append(deepcopy(ledger.snapshot())),
    ) as model:
        await model.get_response()
        await model.get_response()
        with pytest.raises(LunaBudgetExceeded):
            await model.get_response()
    assert [request["max_output_tokens"] for request in calls] == [8, 2]
    assert all(request["reasoning"] == {"effort": effort, "mode": "standard"}
               for request in calls)
    assert all(request["store"] is False and request["service_tier"] == "default"
               for request in calls)
    assert all("temperature" not in request and "top_p" not in request for request in calls)
    assert snapshots[0]["requests"][0]["status"] == "pending"
    assert snapshots[-1]["effective_provider_configuration"] == "matched"


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
    assert result["input_tokens"] is None and result["cost_usd"] is None
    assert result["usage_status"] == "unknown" and result["error"] == "RuntimeError"
    assert "secret-bearing" not in json.dumps(result)


def test_completed_usage_survives_later_unknown_request():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    first, _ = ledger.admit()
    ledger.observe_raw(first, response(inp=5, out=2))
    second, _ = ledger.admit()
    second.update(status="error_usage_unknown", error_type="TimeoutError")
    snapshot = ledger.snapshot()
    assert snapshot["usage_status"] == "unknown"
    assert snapshot["observed_input_tokens"] == 5 and snapshot["observed_output_tokens"] == 2
    with pytest.raises(LunaBudgetExceeded):
        ledger.admit()


def test_missing_cache_or_reasoning_detail_remains_unknown():
    ledger = LunaLedger(LunaContract.from_env(luna_env()))
    record, _ = ledger.admit()
    ledger.observe_raw(record, response(usage={"input_tokens": 2, "output_tokens": 3}))
    assert ledger.snapshot()["observed_cached_tokens"] is None
    assert ledger.snapshot()["observed_reasoning_tokens"] is None
