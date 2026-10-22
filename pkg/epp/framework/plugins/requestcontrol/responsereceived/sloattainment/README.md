# SLO Attainment Reporter

**Type:** `slo-attainment-reporter`
**Interfaces:** `requestcontrol.ResponseHeaderProcessor`, `requestcontrol.ResponseBodyProcessor`
**Stability:** Alpha

Classifies each completed request against its TTFT and TPOT targets and counts the results by priority, prompt-length bucket and prefill/decode path. Attainment ratios, error-budget burn and per-path breach attribution then come directly from PromQL.

## What It Does

At end of stream the plugin resolves the request's target from its served priority and prompt bucket, then classifies it:

| Result | Condition |
|---|---|
| `failed` | Status other than 200, or the stream ended without completing (client disconnect, eviction, error). |
| `compliant` | `TTFT <= ttftMs` and `TPOT <= tpotMs`. |
| `ttft_miss`, `tpot_miss`, `ttft_tpot_miss` | The named target was missed. |

- **TTFT** is measured at the EPP from request receipt to the first response byte, so it includes flow-control queueing, prefill, KV transfer and the first decode step.
- **TPOT** is the per-request mean, `(complete - first token) / (output tokens - 1)`. It is scored only for streamed responses with at least two output tokens; otherwise only TTFT is scored.
- A target of `0` is not scored.
- `x-llm-d-slo-ttft-ms` and `x-llm-d-slo-tpot-ms` request headers tighten the configured target for that request, but never loosen it.

Requests at a priority with no target are not counted. List only the priorities that carry a promise. With `qos-admission`, a request downgraded above its team's commit is served at a lower priority, so it falls outside the committed class's attainment.

**Bucket.** The prompt-length bucket comes from the reported `usage.prompt_tokens`, falling back to the tokenized prompt.

**Path:**
- `remote`: the prefill profile picked an endpoint.
- `local`: the prefill profile ran or was declined without one.
- `aggregated`: no prefill profile exists.

Report attainment per path as well as combined: a healthy aggregate can hide a breaching remote path behind a fast local share.

Requests rejected before dispatch (`429`, `503`) never produce a response and are not counted here. Count availability from `llm_d_epp_request_error_total`, `llm_d_epp_flow_control_requests_total{outcome}` and `llm_d_epp_qos_admission_decisions_total`.

## Configuration

```yaml
plugins:
- type: slo-attainment-reporter
  parameters:
    promptBuckets: [1024, 8192]      # labels: le1024, le8192, gt8192
    prefillProfileName: prefill
    targets:
    - {priority: 100, ttftMs: 2000, tpotMs: 33}
    - {priority: 100, promptBucket: gt8192, ttftMs: 6000, tpotMs: 33}
```

| Field | Default | Description |
|---|---|---|
| `promptBuckets` | none | Strictly increasing upper bounds, in prompt tokens. With none, every request is in bucket `all`. |
| `prefillProfileName` | `prefill` | Scheduling profile that marks a remote prefill. Match the `disagg-profile-handler` profile name. |
| `targets[].priority` | required | Served priority the target applies to. |
| `targets[].promptBucket` | any | Bucket label the target applies to. A bucket-specific target overrides the priority-wide one. |
| `targets[].ttftMs` | `0` | TTFT target in milliseconds. |
| `targets[].tpotMs` | `0` | Mean TPOT target in milliseconds. |

## Observability

| Metric | Type | Labels |
|---|---|---|
| `llm_d_epp_slo_requests_total` | Counter | `target_model_name`, `fairness_id`, `priority`, `prompt_bucket`, `path`, `result` |
| `llm_d_epp_slo_ttft_seconds` | Histogram | `priority`, `prompt_bucket`, `path` |
| `llm_d_epp_slo_tpot_seconds` | Histogram | `priority`, `path` |

The histograms record successful scored requests only.

### PromQL

Attainment ratio `R_SLA` per bucket and path, over 30 days:

```promql
sum by (priority, prompt_bucket, path) (increase(llm_d_epp_slo_requests_total{result="compliant"}[30d]))
/
sum by (priority, prompt_bucket, path) (increase(llm_d_epp_slo_requests_total[30d]))
```

`R_TTFT` counts `result=~"compliant|tpot_miss"` in the numerator, and `R_TPOT` counts `result=~"compliant|ttft_miss"`.

Error-budget burn rate against a 99% target, over 1 hour:

```promql
(1 - (
  sum(rate(llm_d_epp_slo_requests_total{priority="100",result="compliant"}[1h]))
  / sum(rate(llm_d_epp_slo_requests_total{priority="100"}[1h]))
)) / (1 - 0.99)
```

p95 TTFT per bucket and path:

```promql
histogram_quantile(0.95, sum by (le, prompt_bucket, path) (rate(llm_d_epp_slo_ttft_seconds_bucket{priority="100"}[5m])))
```

When the remote path breaches, split it by stage using the sidecar's `llm_d_disagg_sidecar_prefill_duration_seconds` and the KV-transfer metrics.
