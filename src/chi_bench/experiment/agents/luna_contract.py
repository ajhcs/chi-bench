"""Fail-closed Luna request contract and durable accounting; no SDK imports.

Every admitted request reserves the official context ceiling for input tokens.
Reservations are never reclaimed. Actual raw Responses metadata is checked before
SDK conversion. This is not a dollar budget or permission to run a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

LUNA_MODEL = "gpt-6-luna"
LUNA_ENDPOINT = "https://api.openai.com/v1"
# https://developers.openai.com/api/docs/models/gpt-6-luna (2026-10-20)
LUNA_CONTEXT_TOKENS = 1_050_000
LUNA_MAX_OUTPUT_TOKENS = 128_000


class LunaContractError(ValueError):
    """The requested route cannot satisfy the declared evaluation contract."""


class LunaBudgetExceeded(RuntimeError):
    """No further model request is admitted for this trial."""


@dataclass(frozen=True)
class LunaContract:
    effort: str
    mode: str
    max_output_tokens: int
    max_total_output_tokens: int
    max_total_input_tokens: int
    max_model_calls: int
    max_total_tokens: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LunaContract:
        if env.get("OPENAI_AGENTS_MODEL") != LUNA_MODEL:
            raise LunaContractError("Luna requires the exact bare gpt-6-luna model ID")
        if env.get("OPENAI_BASE_URL", "").rstrip("/") != LUNA_ENDPOINT:
            raise LunaContractError("Luna requires the direct OpenAI endpoint")
        if env.get("OPENAI_AGENTS_API_MODE") != "responses":
            raise LunaContractError("Luna tool evaluation requires explicit Responses mode")
        effort = env.get("OPENAI_AGENTS_REASONING_EFFORT", "")
        mode = env.get("OPENAI_AGENTS_REASONING_MODE", "")
        if effort not in {"medium", "max"} or mode not in {"standard", "pro"}:
            raise LunaContractError("Declare Luna medium/max and standard/pro explicitly")
        if mode == "pro":
            raise LunaContractError(
                "Pro work aggregation has no verified context-ceiling bound; no request admitted"
            )
        if env.get("OPENAI_AGENTS_MAX_RETRIES") != "0":
            raise LunaContractError("Luna budget accounting requires retries explicitly disabled")
        limits = {}
        for name in (
            "max_output_tokens", "max_total_output_tokens", "max_total_input_tokens",
            "max_model_calls", "max_total_tokens"
        ):
            raw = env.get(f"OPENAI_AGENTS_{name.upper()}", "")
            if not raw.isdecimal() or int(raw) < 1:
                raise LunaContractError(f"Declare a positive integer {name}")
            limits[name] = int(raw)
        if limits["max_output_tokens"] > LUNA_MAX_OUTPUT_TOKENS:
            raise LunaContractError("Per-call output exceeds the official Luna output limit")
        return cls(effort=effort, mode=mode, **limits)


@dataclass
class LunaLedger:
    contract: LunaContract
    requests: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = 0
    reasoning_tokens: int | None = 0
    blocked_reason: str | None = None

    @property
    def complete(self) -> bool:
        return all(r["status"] == "complete" for r in self.requests)

    def admit(self) -> tuple[dict[str, Any], int]:
        if not self.complete or self.blocked_reason:
            raise LunaBudgetExceeded("Usage is unknown or the trial budget is exhausted")
        c = self.contract
        if c.mode != "standard":
            raise LunaContractError("Only standard mode has a context-ceiling admission contract")
        reserved_input = sum(r["reserved_input_tokens"] for r in self.requests)
        reserved_output = sum(r["max_output_tokens"] for r in self.requests)
        if len(self.requests) >= c.max_model_calls:
            self.blocked_reason = "call_budget_exhausted"
        elif reserved_input + LUNA_CONTEXT_TOKENS > c.max_total_input_tokens:
            self.blocked_reason = "input_reservation_cannot_admit_request"
        if self.blocked_reason:
            raise LunaBudgetExceeded(self.blocked_reason)
        remaining_total = c.max_total_tokens - reserved_input - reserved_output - LUNA_CONTEXT_TOKENS
        limit = min(c.max_output_tokens, c.max_total_output_tokens - reserved_output, remaining_total)
        if limit <= 0:
            self.blocked_reason = "output_or_total_reservation_cannot_admit_request"
            raise LunaBudgetExceeded(self.blocked_reason)
        record = {
            "call": len(self.requests) + 1,
            "status": "pending",
            "reserved_input_tokens": LUNA_CONTEXT_TOKENS,
            "max_output_tokens": limit,
            "provider_contract": "unverified",
        }
        self.requests.append(record)
        return record, limit

    def observe_raw(self, record: dict[str, Any], response: Any) -> None:
        """Inspect the actual Responses object, never the reduced SDK ModelResponse."""
        self.raw_responses.append(response)
        raw = response.model_dump(mode="json")
        record["raw_provider_receipt"] = {
            key: raw.get(key) for key in (
                "id", "model", "status", "reasoning", "service_tier", "usage", "incomplete_details"
            )
        }
        record["response_id"] = raw.get("id")
        record["request_id"] = getattr(response, "_request_id", None)
        usage = raw.get("usage") or {}
        values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
        if any(type(value) is not int or value < 0 for value in values):
            record["status"] = "usage_unknown"
            raise LunaContractError("Luna response omitted usable token accounting")
        inp, out = values
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        if (any(value is not None and (type(value) is not int or value < 0)
                for value in (cached, reasoning)) or
                (cached is not None and cached > inp) or
                (reasoning is not None and reasoning > out)):
            record["status"] = "usage_unknown"
            raise LunaContractError("Luna response contained invalid usage detail")
        record.update(
            status="complete", input_tokens=inp, output_tokens=out,
            cached_tokens=cached, reasoning_tokens=reasoning,
        )
        self.input_tokens += inp
        self.output_tokens += out
        self.cached_tokens = (self.cached_tokens + cached
                              if self.cached_tokens is not None and cached is not None else None)
        self.reasoning_tokens = (self.reasoning_tokens + reasoning
                                 if self.reasoning_tokens is not None and reasoning is not None
                                 else None)
        effective = raw.get("reasoning") or {}
        if raw.get("model") != LUNA_MODEL:
            self.blocked_reason = "returned_model_mismatch_or_absent"
        elif effective.get("effort") != self.contract.effort or effective.get("mode") != self.contract.mode:
            self.blocked_reason = "returned_reasoning_mismatch_or_absent"
        elif raw.get("service_tier") != "default":
            self.blocked_reason = "returned_processing_tier_mismatch_or_absent"
        elif raw.get("status") != "completed":
            self.blocked_reason = "provider_response_not_completed"
        elif inp > record["reserved_input_tokens"] or out > record["max_output_tokens"]:
            self.blocked_reason = "provider_exceeded_token_reservation"
        if self.blocked_reason:
            record["provider_contract"] = "invalid"
            raise LunaContractError(self.blocked_reason)
        record["provider_contract"] = "matched"

    def snapshot(self) -> dict[str, Any]:
        return {
            "model": LUNA_MODEL,
            "endpoint": LUNA_ENDPOINT,
            "reasoning": {"effort": self.contract.effort, "mode": self.contract.mode},
            "limits": vars(self.contract),
            "requests": self.requests,
            "usage_status": "complete" if self.complete else "unknown",
            "observed_input_tokens": self.input_tokens,
            "observed_output_tokens": self.output_tokens,
            "observed_cached_tokens": self.cached_tokens,
            "observed_reasoning_tokens": self.reasoning_tokens,
            "blocked_reason": self.blocked_reason,
            "reserved_input_tokens": sum(r["reserved_input_tokens"] for r in self.requests),
            "reserved_output_tokens": sum(r["max_output_tokens"] for r in self.requests),
            "effective_provider_configuration": (
                "matched" if self.requests and all(r["provider_contract"] == "matched"
                                                   for r in self.requests) else "unverified_or_invalid"
            ),
            "cost_usd": None,
            "cost_status": "unknown",
        }


class LunaResponsesBoundary:
    """Public responses.create boundary with no SDK-private request inspection."""

    def __init__(self, responses, ledger: LunaLedger, persist):
        self.responses = responses
        self.ledger = ledger
        self.persist = persist

    async def create(self, **kwargs):
        import time

        contract = self.ledger.contract
        if kwargs.get("model") != LUNA_MODEL or kwargs.get("stream"):
            raise LunaContractError("Exact non-streaming Luna Responses request required")
        requested_reasoning = kwargs.get("reasoning") or {}
        if requested_reasoning.get("effort") != contract.effort:
            raise LunaContractError("SDK changed the requested Luna effort")
        if requested_reasoning.get("mode", contract.mode) != contract.mode:
            raise LunaContractError("SDK changed the requested Luna mode")
        # Additional body/query data must not override the enforced wire contract.
        for key in ("extra_body", "extra_query", "previous_response_id", "conversation", "prompt"):
            if kwargs.pop(key, None):
                raise LunaContractError(f"Unsupported Luna wire override: {key}")
        for key in ("temperature", "top_p", "top_logprobs"):
            value = kwargs.pop(key, None)
            if type(value) in (int, float):
                raise LunaContractError("Sampling parameters are outside the Luna contract")
        record, limit = self.ledger.admit()
        kwargs.update(
            max_output_tokens=limit,
            reasoning={"effort": contract.effort, "mode": contract.mode},
            service_tier="default",
            store=False,
        )
        record["requested_configuration"] = {
            "model": kwargs["model"], "max_output_tokens": limit,
            "reasoning": kwargs["reasoning"], "service_tier": "default", "store": False,
        }
        self.persist()  # Before transport: a hard kill leaves an uncertain pending request.
        started = time.monotonic()
        try:
            response = await self.responses.create(**kwargs)
            self.ledger.observe_raw(record, response)
            return response
        except BaseException as exc:
            if record["status"] == "pending":
                record.update(status="error_usage_unknown", error_type=type(exc).__name__)
            raise
        finally:
            record["latency_seconds"] = time.monotonic() - started
            self.persist()
