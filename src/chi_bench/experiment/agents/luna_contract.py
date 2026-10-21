"""Fail-closed Luna request contract and durable accounting; no SDK imports.

Output tokens include reasoning. The total-token guard is checked after each
response; it cannot bound unknown input usage or ambiguous failed requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

LUNA_MODEL = "gpt-6-luna"
LUNA_ENDPOINT = "https://api.openai.com/v1"


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
        if env.get("OPENAI_AGENTS_MAX_RETRIES") != "0":
            raise LunaContractError("Luna budget accounting requires retries explicitly disabled")
        limits = {}
        for name in (
            "max_output_tokens", "max_total_output_tokens", "max_model_calls", "max_total_tokens"
        ):
            raw = env.get(f"OPENAI_AGENTS_{name.upper()}", "")
            if not raw.isdecimal() or int(raw) < 1:
                raise LunaContractError(f"Declare a positive integer {name}")
            limits[name] = int(raw)
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
        remaining = c.max_total_output_tokens - self.output_tokens
        if len(self.requests) >= c.max_model_calls or remaining <= 0:
            self.blocked_reason = "call_or_output_budget_exhausted"
            raise LunaBudgetExceeded(self.blocked_reason)
        if self.input_tokens + self.output_tokens >= c.max_total_tokens:
            self.blocked_reason = "observed_total_token_guard_exhausted"
            raise LunaBudgetExceeded(self.blocked_reason)
        limit = min(c.max_output_tokens, remaining,
                    c.max_total_tokens - self.input_tokens - self.output_tokens)
        record = {"call": len(self.requests) + 1, "status": "pending", "max_output_tokens": limit}
        self.requests.append(record)
        return record, limit

    def observe(self, record: dict[str, Any], response: Any) -> None:
        self.raw_responses.append(response)
        usage = getattr(response, "usage", None)
        values = [getattr(usage, k, None) for k in ("input_tokens", "output_tokens")]
        record["response_id"] = getattr(response, "response_id", None)
        record["request_id"] = getattr(response, "request_id", None)
        if any(type(v) is not int or v < 0 for v in values):
            record["status"] = "usage_unknown"
            raise LunaBudgetExceeded("Luna response omitted usable token accounting")
        inp, out = values
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        output_details = getattr(usage, "output_tokens_details", None)
        reasoning = getattr(output_details, "reasoning_tokens", None)
        if any(v is not None and (type(v) is not int or v < 0)
               for v in (cached, reasoning)):
            record["status"] = "usage_unknown"
            raise LunaBudgetExceeded("Luna response contained invalid usage detail")
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
        if out > record["max_output_tokens"]:
            self.blocked_reason = "provider_output_limit_exceeded"
        elif self.input_tokens + self.output_tokens > self.contract.max_total_tokens:
            self.blocked_reason = "observed_total_token_guard_exceeded"
        if self.blocked_reason:
            raise LunaBudgetExceeded(self.blocked_reason)

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
            "effective_provider_configuration": "unverified",
            "cost_usd": None,
            "cost_status": "unknown",
        }
