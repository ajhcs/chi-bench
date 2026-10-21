# Luna evaluation contract (source preparation)

This fork adds an opt-in contract for exactly `gpt-6-luna` at
`https://api.openai.com/v1/responses`. It does not establish SDK compatibility,
provider access, model quality, benchmark validity or authorization to execute.
No model calls or benchmark trials accompanied this source change.

The OpenAI Agents harness accepts `openai/gpt-6-luna` or `gpt-6-luna`, strips
only the declared vendor prefix, and rejects alternate endpoints/provider routes.
The runner requires the bare ID, explicit Responses API, effort `medium` or
`max`, mode `standard` or `pro`, and retries explicitly set to zero. Unsupported
values fail; no substitution to another effort, model or provider occurs.

Official documentation consulted October 20, 2026:
[model identity](https://developers.openai.com/api/docs/models/gpt-6-luna) and
[reasoning mode](https://developers.openai.com/api/docs/guides/reasoning).
`reasoning.mode` and `reasoning.effort` are independent request fields; pro mode
keeps the model ID. The adapter passes both fields unchanged. The pinned Agents
SDK 0.13.6 source exposes `ModelSettings.reasoning` and maps `max_tokens` to
Responses `max_output_tokens`; transport and server acceptance remain untested.

## Explicit agent kwargs

The following is a configuration excerpt, not execution approval:

```yaml
agent: openai-agents
model: openai/gpt-6-luna
agent_kwargs:
  api_mode: responses
  reasoning_effort: medium
  reasoning_mode: standard
  max_retries: "0"
  max_output_tokens: "1000"
  max_total_output_tokens: "3000"
  max_model_calls: "3"
  max_total_tokens: "10000"
```

Change only `reasoning_effort` to `max` for the paired arm. Keep mode, tools,
versions, task bytes, reset state, budgets, timeouts, skill condition, simulator
and judge fixed. These illustrative limits are not measured CHI requirements.
Handbook access and real task selection remain separately gated. The contract
bounds Luna calls only; it does not budget the judge or conversation simulators.

## Budget and accounting boundaries

Before every request, the adapter journals a pending entry and reserves the
smaller of the per-call output limit and remaining aggregate output allowance.
It also limits model call count. The explicit SDK client and runner both disable
retries. Pending/error/missing-usage entries prohibit another model request.
Streaming is unsupported by this bounded route.

The aggregate total-token guard is checked against observed input plus output
usage after each response. The next request's input usage is unknown beforehand,
so one response can exceed that guard. It is not a hard total-token or USD cap.
Reasoning is included in output usage. Provider enforcement of output limits and
actual billing remain unverified. A strict billing limit requires separate
provider controls, current pricing and an approved execution plan.

`agent/luna_usage.json` is replaced atomically before and after calls. It records
requested identity/effort/mode, limits, call status, response/request IDs when
available, observed tokens, latency and budget termination. Effective provider
configuration is explicitly unverified: the pinned SDK's generic response does
not preserve every raw configuration field. Missing cache/reasoning details
remain null. A killed process may leave a pending entry; do not retry it as free.

`agent/run_result.json` uses null totals and `usage_status: unknown` when usage
cannot be recovered. An exception after completed Luna calls preserves their
observed ledger entries. Raw error text is not copied into the result artifact;
only its exception class is stored. Luna cost is deliberately null/unknown until
complete billing evidence and a reviewed pricing implementation exist. Generic
harness costs retain their existing estimated status.

The aggregator refuses runs with an invalid-judge scorecard. It retains valid
outcomes when new accounting evidence is unknown, emits null mean cost, and
reports `n_unknown_cost_trials`; it never interprets those entries as free.
Historical artifacts without the new validity/accounting evidence keep legacy
behavior and must not be treated as newly validated evidence.

## Required judge validity

`apply_judge_validity` retains `upstream_binary_reward`, marks missing required
rubrics or judge outages as `evaluation_validity: invalid`, and writes numeric
reward zero for Harbor's scalar interface. Invalid grading is not an agent
failure: verifier exit code 2 and the validity metadata distinguish it. Rejudge
before aggregation. Legitimate `no_submit_e2e_path` skips remain applicable NA;
a genuine negative judge verdict remains a valid failed trial.

## Validation status and remaining gates

Mocked tests cover effort/mode/routing, SDK boundary settings, output reservation,
call and observed-token guards, unknown usage, judge invalidity and intentional
NA. They are prepared, not executed. Only owned static AST parsing and Git diff
inspection are permitted during source preparation. No dependency, workflow,
security, credential, service or dataset settings are changed.

Before runtime acceptance: independent review of the immutable commit; approval
for isolated dependencies and offline mocked tests; SDK wire-contract validation;
resolved immutable fixture/access/repair hashes; approved image/resources/network;
complete judge/simulator budgets; and separately approved, estimated billed smoke.
No merge or benchmark publication is authorized by this document.
