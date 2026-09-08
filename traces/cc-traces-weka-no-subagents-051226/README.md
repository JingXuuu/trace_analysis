---
license: apache-2.0
pretty_name: CC Traces — Weka, No-Subagents (May 12 2026)
task_categories:
  - text-generation
tags:
  - llm
  - inference
  - benchmarking
  - kv-cache
  - agentic
  - multi-turn
  - claude
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: traces.jsonl
---

# CC Traces — Weka, No-Subagents (May 12 2026)

A collection of **949 multi-turn agentic traces** (≈ 136.1 k individual
model requests) drawn from real production traffic against the Claude
Code CLI ≥ 2.1.139. Each trace captures the full request/response
sequence of a single agent session, including per-request KV block
hashes, so the dataset can be replayed against an inference engine or
used to simulate prefix-cache behavior offline.

**No-subagents variant.** This dataset is a derivative of the source
weka traces with all `WekaSubagentEntry` blocks stripped — only
top-level main-agent turns remain. The full-subagent companion variant
lives elsewhere. Use this corpus when you want a single
linear agent stream per trace and don't care about the parent / child
fan-out structure of agentic tool-calling.

- **Traces:** 949
- **Requests:** 136,118 total, mean **143.4** per trace, max **13,685**
- **Models:** `claude-opus-4-7` (most turns), `claude-haiku-4-5-20251001`,
  `claude-opus-4-6`, `claude-sonnet-4-6`
- **KV block size:** 64 tokens
- **Hash scope:** `local` — block hash IDs are only comparable *within*
  a single trace; they are not a global content-addressable identity.

## Important: tokenizer caveat

The `in` field on each request and the `hash_ids` array are both
measured in **the proxy's tokenizer** (qwen3 / o200k_base — depends on
the originating trace_version), NOT Anthropic's BPE. Anthropic
typically reports ~60 % of the qwen3/o200k token count for the same
content. So the ISL numbers below are larger than what the Anthropic
API would have billed for the same prompt — but they're self-consistent
between `in` and `hash_ids`, which is what matters for KV-cache replay
simulation.

If you replay these traces against a real Claude (or Claude-like)
server, the server will re-tokenize the text and see ~Claude-equivalent
token counts, so cache-hit-rate measurements remain accurate.

## What's in each trace

Top-level trace fields:

| field           | type             | description                                         |
|-----------------|------------------|-----------------------------------------------------|
| `id`            | str              | Trace identifier (the proxy session id)             |
| `models`        | list[str]        | Models used by the trace                            |
| `block_size`    | int              | KV block size used to derive `hash_ids` (64)        |
| `hash_id_scope` | str              | `local` — hash IDs are per-trace, not global        |
| `requests`      | list[object]     | Ordered list of per-request records (main-agent only) |

Each entry in `requests` is:

| field         | type      | description                                                              |
|---------------|-----------|--------------------------------------------------------------------------|
| `t`           | float     | Seconds since start of trace                                             |
| `type`        | str       | `n` (non-streaming) or `s` (streaming)                                   |
| `model`       | str       | Target model for this request                                            |
| `in`          | int       | Effective prompt tokens covered by `hash_ids` (proxy tokenizer)          |
| `out`         | int       | Output tokens (Anthropic-reported)                                       |
| `hash_ids`    | list[int] | Per-trace local block-hash IDs for the input, one per 64-token block     |
| `api_time`    | float     | End-to-end server time for this call (seconds)                           |
| `think_time`  | float     | Wall-clock gap from previous request's end (seconds)                     |
| `ttft`        | float?    | Time to first token (streaming requests only)                            |

`hash_ids` make the dataset unusually useful for KV-cache work: the
contiguous common prefix between turn *t* and turn *t − 1* exactly
measures the portion of the input that a local prefix cache would be
able to reuse.

## Summary statistics

|                              |   p50    |   p75    |   p90    |   p95    |   p99    |   mean   |
|------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| ISL (tokens)                 |  123,952 |  245,124 |  391,085 |  490,478 |  720,485 |  178,018 |
| OSL (tokens)                 |      261 |      664 |    1,614 |    2,805 |    7,013 |      711 |
| `think_time` (s)             |     1.29 |     2.92 |    54.97 |   181.78 | 1,887.83 |   351.34 |
| Turn depth (requests/trace)  |       56 |      118 |      252 |      437 |    1,497 |      143 |
| Token growth per turn        |      395 |      952 |    2,504 |    5,399 |   87,849 |      758 |
| Cache hit rate per request   |   0.8404 |   0.9243 |   0.9678 |   0.9970 |   0.9995 |   0.7509 |

### Why is "token growth per turn" sometimes negative?

Agentic prompts usually grow monotonically — each turn appends the
user's new message + assistant's reply to the running context, so the
next turn's `in` is at least as big as the previous. **But ~1% of
adjacent-turn deltas in this corpus are sharply negative** (Δin
< −100K tokens, often −500K to −800K). These are real production
events, not data errors: they're **context compaction**.

Claude Code (and similar agentic clients) summarize long conversations
when the context grows close to the model's limit, replacing dozens of
prior messages with a short summary. After that summary is committed,
the next turn's prompt drops from ~800K back to ~10–80K tokens — a
sharp negative delta. Some sessions in this corpus compact 10+ times
in a row, which produces clusters of large negative growth events
that are visible in the histogram's left tail.

If you're modeling raw prompt growth, treat negative deltas as
"context reset" events and either skip them or count them separately.
If you're modeling realistic agentic load, leave them in — they're
part of how long sessions stay under the context budget.

Mean per-request prefix-cache hit rate across the full corpus is
**~75 %**, with a strong mode near 1.0 (steady-state cache reuse) and a
secondary mass near 0 (cold-start / reset turns).

## Plots

Combined 3×2 distributions with p50 / p75 / p90 / p99 percentile
markers on each panel. Two variants: log-x for the heavy-tailed
metrics, linear-x for the bounded ones.

### Log-x

Best for the long-tailed distributions (think_time, ISL, OSL, turn
depth). Token growth stays linear (it has negative values from
context-compaction events).

![Distributions — log x](plots/distributions_log.png)

### Linear-x

Linear axes everywhere. Heavy tails clipped at p99 with a count of
clipped values in each panel's subtitle so they're not silently
dropped.

![Distributions — linear x](plots/distributions_linear.png)

## How this dataset was built

1. **Sampled** from the SemiAnalysis Claude Code proxy database.
   Filtered to: Anthropic models only (`model LIKE 'claude-%'`), HTTP
   200 + no proxy error, `privacy_mode = 'anon'` (request bodies are
   redacted; metric columns are intact), and the post-2026-04-16
   timeframe (after the `subagent_label` migration so subagent
   classification is reliable).
2. **Converted** flat per-row dumps into v1 weka format using the
   conversation-view subagent grouping algorithm from the SemiAnalysis
   claude-code-proxy. Hash IDs were remapped to per-trace local ints
   (`hash_id_scope: "local"`).
3. **Subagent entries removed.** This is the no-subagents variant of
   the corpus, so every `WekaSubagentEntry` block was filtered out of
   each trace's `requests` list. Traces that became empty after
   filtering (entirely Agent-SDK or utility-only sessions) were
   dropped.
4. **Proxy-hash-bug outliers removed.** Sessions containing any
   single row with `in > 1,000,000` proxy-tokens were dropped (25
   sessions). These are confirmed proxy hashing artifacts (v1/v2/v4
   over-count `hash_token_count` for sessions with document content
   blocks or pre-v3 tool-call histories — not legitimate token sizes
   that the model actually saw).

## Intended uses

- **Inference benchmarking:** replay traces against a serving engine
  (vLLM, SGLang, TRT-LLM, etc.) to measure throughput / latency under a
  realistic agentic workload.
- **KV-cache research:** the `hash_ids` field exposes block-level
  prefix structure directly without needing to re-tokenize any text.
- **Capacity planning:** the ISL / OSL / think-time / turn-depth
  distributions are a reasonable first-order model of traffic from
  long-context agentic clients.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("semianalysisai/cc-traces-weka-no-subagents-051226", split="train")
print(ds[0]["id"], len(ds[0]["requests"]))
```

## License & attribution

Released under the **Apache 2.0** license.
