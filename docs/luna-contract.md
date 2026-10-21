# Luna evaluation contract (source preparation)

This fork adds an opt-in contract for exactly `gpt-6-luna` at
`https://api.openai.com/v1/responses`. It does not establish SDK compatibility,
provider access, model quality, benchmark validity or authorization to execute.
No model calls or benchmark trials accompanied this source change.

The OpenAI Agents harness accepts `openai/gpt-6-luna` or `gpt-6-luna`, strips
only the declared vendor prefix, and rejects alternate endpoints/provider routes.
The runner requires the bare ID, explicit Responses API, effort `medium` or
`max`, explicit `standard` mode, and retries set to zero. The public mode flag
recognizes `pro`, but this conservative reservation route rejects it before any
request: aggregated pro work has no verified context-sized usage bound here.
Unsupported values fail; no substitution of effort, mode, model or provider occurs.

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
  max_total_input_tokens: "3150000"
  max_model_calls: "3"
  max_total_tokens: "3153000"
```

Change only `reasoning_effort` to `max` for the paired arm. Keep mode, tools,
versions, task bytes, reset state, budgets, timeouts, skill condition, simulator
and judge fixed. These illustrative limits are not measured CHI requirements.
Handbook access and real task selection remain separately gated. The contract
bounds Luna calls only; it does not budget the judge or conversation simulators.

## Budget and accounting boundaries

Before every generation request, the adapter reserves the full official
1,050,000-token Luna context ceiling for input, plus the admitted output cap.
`max_total_input_tokens` and `max_total_tokens` must have room for that reservation
before transport is called. Input and output reservations are never reclaimed,
even when actual returned usage is smaller. The next output cap is the minimum
of the per-call cap, remaining aggregate output allowance and remaining aggregate
total allowance after the context reservation. Model call count is capped too.
An insufficient input/total/output allowance refuses the request before sending.
The context ceiling is a conservative documented provider bound, not an estimate
of task input. Its continued validity must be checked before execution.

The journal records the pending reservation before the public SDK
`responses.create` call. Both SDK and runner retries are disabled, including
retry-disabled clients derived through `with_options`. Pending/error/missing-usage
entries prohibit another model request. Streaming and extra body/query overrides
are unsupported. The wire contract forces Standard processing (`service_tier:
default`), `store: false`, the exact model, requested effort and standard reasoning
mode. No sampling parameters are sent.

Returned usage exceeding a reserved limit is a provider contract failure. Reasoning
is included in output usage. This is not a USD cap: context-sized reservations can
cross long-context pricing thresholds, and judge/simulator costs remain separate.
A billing limit requires current rates, provider controls and an approved plan.

`agent/luna_usage.json` is file/directory-fsynced and replaced atomically before and after calls. It records
requested identity/effort/mode, reservations, limits, call status, response/request
IDs when available, observed tokens, latency and budget termination. A boundary
around the public `responses.create` API preserves raw returned model, reasoning,
processing tier, status and usage before the SDK reduces them to `ModelResponse`.
Missing/mismatched model, effort, mode or processing tier, or a non-completed raw
status, stops the run before SDK consumers can execute further tools. Matching
these returned fields is recorded as such; it does not establish endpoint access,
wire compatibility or benchmark acceptance without the pending runtime checks.
Missing cache/reasoning details remain null. A killed process may leave a pending
entry; do not retry it as free.

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
call and pre-generation context reservations, raw returned identity/status,
unknown usage, derived-client confinement, judge invalidity and intentional
NA. They are prepared, not executed. Only owned static AST parsing and Git diff
inspection are permitted during source preparation. No dependency, workflow,
security, credential, service or dataset settings are changed.

Before runtime acceptance: independent review of the immutable commit; approval
for isolated dependencies and offline mocked tests; SDK wire-contract validation;
resolved immutable fixture/access/repair hashes; approved image/resources/network;
complete judge/simulator budgets; and separately approved, estimated billed smoke.
No merge or benchmark publication is authorized by this document.

## Source publication without research execution

For this change, the fork main base is
`4444a07fa68a4f1623f841a292a54e284fc8e7f7`. Its complete `.github` tree and
the source branch contain only `.github/workflows/ci.yml`, blob
`25369ba2185905e5ec72110895a9c574f77e145a`, with triggers `[push, pull_request]`.
There is no `pull_request_target`, dispatch or chained workflow in those trees.
The correction commit uses the supported transparent `[skip ci]` instruction to
publish source without executing dependency installation, tests or Docker builds.
[GitHub documents this behavior](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).
Skipped checks remain unrun/Pending and provide no validation. No workflow or
security settings or Git hooks are changed; no merge is authorized. Recheck
base/branch workflows before any further publication if either tree changes.
