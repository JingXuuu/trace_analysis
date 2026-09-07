---
license: mit
task_categories:
- text-generation
tags:
- kv-cache
- llm-serving
- agentic
- multi-turn
- traces
- benchmark
pretty_name: LMCache Agentic Dataset Collection
size_categories:
- 10K<n<100K
---

# LMCache Agentic Dataset Collection

A curated dataset collection of **787 multi-turn agentic LLM sessions** (24,881 total LLM iterations) designed for benchmarking stateful LLM serving systems. Every session exhibits at least 5 turns with prefix growth and builds to at least 10K tokens of context — making it ideal for evaluating tiered KV Cache solutions like [LMCache](https://github.com/LMCache/LMCache).

## Motivation

Modern LLM agents (coding assistants, research agents, tool-calling systems) make dozens of sequential API calls per task. Each call appends tool results and assistant responses to a growing conversation history, creating a natural prefix-sharing pattern: iteration N's input is iteration N-1's input plus new messages at the end.

This means **>90% of tokens in a typical request have already been processed in the previous request**. An efficient KV cache can skip recomputation of the shared prefix, dramatically increasing system throughput (tok/s) and reducing GPU cost. 

The challenge with inference benchmarking of agentic workloads is that the trace format should permit directly running against an inference API such as the OpenAI or Anthropic API instead of requiring an agent harness to be nested in between. Thus, the user can directly benchmark their inference deployment with a reproducable agentic workload without actually deploying an agent.

This dataset provides these real agent trajectories from the tasks provided in the following open source agentic datasets:

- **[SWE-bench Verified](https://swebench.com/)**: Real GitHub issues from popular Python repos. The agent (OpenHands CodeAct) reads code, writes patches, runs tests, and iterates on failures. Sessions range from 5-50 turns of edit-test-debug cycles.
- **[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)**: Level 2-3 multi-step reasoning tasks requiring web search, file analysis, and chain-of-thought reasoning. Evaluated with Inspect AI.
- **[WildClaw](https://github.com/WildClaw)**: Mixed agent tasks spanning creative synthesis, search, and code generation.

## Dataset Overview

| Source | Sessions | Turns (med/mean/max) | Models | Workload |
|--------|----------|----------------------|--------|----------|
| **SWE-bench** | 669 | 38 / 35 / 50 | MiniMax-M2.5, Claude Sonnet 4.6, DeepSeek V3.1 | Code debugging: read code, write patches, run tests, debug failures |
| **GAIA** | 85 | 14 / 13 / 26 | Claude Sonnet 4.6 | Multi-step research and reasoning with web search |
| **WildClaw** | 10 | 22 / 24 / 41 | Claude Opus 4.6 | Mixed agent tasks (creative, search, code) |
| **Total** | **787** | **35 / 33 / 50** | **4 models** | **3 workload types** |

## Data Format

Each row represents one LLM iteration within a session. All rows for a session share the same `session_id`. A "session" corresponds to a single task — one initial query that the agent works on across multiple turns. All sessions are single-task: the agent receives one problem and iterates on it (reading code, running commands, calling tools) until done or the turn budget is exhausted.

Messages use four roles: `system` (always the first message), `user` (the initial query plus framework-injected feedback like error messages and runtime info), `assistant` (LLM responses and tool-call requests), and `tool` (tool execution results). The `tool` role appears in GAIA, WildClaw, and SWE-bench Sonnet sessions which use OpenAI-style function calling. SWE-bench MiniMax/DeepSeek sessions embed tool outputs directly in `user` messages instead.

```jsonl
{"session_id": "swebench__django__django-16527__claude", "model": "claude-sonnet-4-6", "input": [{"role": "system", "content": "..."}, {"role": "user", "content": "Fix the bug..."}], "pre_gap": 0.0, "output_length": 342}
{"session_id": "swebench__django__django-16527__claude", "model": "claude-sonnet-4-6", "input": [{"role": "system", "content": "..."}, {"role": "user", "content": "Fix the bug..."}, {"role": "assistant", "content": "Let me read...", "tool_calls": [...]}, {"role": "tool", "content": "class QuerySet:..."}], "pre_gap": 1.2345, "output_length": 567}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier (e.g. `swebench__django__django-16527__claude`, `gaia__L2_abc123__claude`, `wildclaw__01_Productivity_Flow_task_1_arxiv_digest__claude`). Describes source, task, and model. Same for all rows in a session. |
| `model` | string | Model used (e.g. `claude-sonnet-4-6`, `minimax-m2.5`, `deepseek-v3.1`). |
| `input` | array | Full cumulative OpenAI-format `messages` array. The assistant's output from iteration N-1 is embedded in iteration N's input. `input[N]` is a strict prefix-superset of `input[N-1]`. |
| `pre_gap` | float | Seconds between the previous iteration's response completing (last streamed token) and this iteration's request being sent. This is the real tool-execution / user-thinking time. Always `0.0` for the first iteration of a session. Median: 0.71s, mean: 2.08s, p95: 3.73s. |
| `output_length` | int | Completion tokens generated for this iteration. |

### Timing Model

The `pre_gap` field captures the **client-side delay** between consecutive LLM calls within a session. This is the time spent executing tools (running bash commands, reading files, making web requests) or processing the LLM's response before sending the next request. It does **not** include LLM inference time — that depends on the serving system being benchmarked.

```
|-- LLM inference (N-1) --|-- pre_gap[N] (tool exec / think time) --|-- LLM inference (N) --|
```

This enables accurate trace replay: a benchmarking tool can fire request N exactly `pre_gap[N]` seconds after receiving the last token of response N-1, faithfully reproducing the original workload timing without baking in the original server's inference latency.

## Dataset Statistics

### Turns per Session

The dataset spans a wide range of conversation lengths. SWE-bench sessions are the longest (median 38 turns, many hitting the 50-turn cap). GAIA sessions vary from 5 to 26 turns depending on task difficulty. WildClaw sessions range from 6 to 41 turns.

![Turns per session distribution](stats/turns_per_session.png)

![Turns per session by source](stats/turns_by_source.png)

### Context Growth

The key property for KV cache benchmarking: how context size grows as a session progresses. All sessions are filtered to reach at least 10K tokens of context. On average, context starts at ~14K tokens and grows linearly to ~35K tokens by turn 50.

![Context growth curve](stats/context_growth.png)

Per-source context growth shows distinct patterns:
- **SWE-bench**: steady linear growth from ~14K to ~37K tokens (large system prompts + code context)
- **GAIA**: rapid growth to ~20K tokens then plateau (web search results accumulate then stabilize)
- **WildClaw**: steep growth from ~30K to ~130K tokens (complex multi-tool agent sessions)

![Context growth by source](stats/context_growth_by_source.png)

### Token Distributions

Input tokens (prompt size) follow a right-skewed distribution with median 21K tokens, dominated by SWE-bench's large contexts. Output tokens have a median of 104 tokens — these are the short, frequent tool-call requests that dominate agentic iteration. However, the dataset also captures substantial long-form generation: 11.5% of outputs exceed 500 tokens and the tail extends to 11K+ tokens. These longer outputs — code patches, detailed analyses, multi-step plans — are the responses typically visible to the end user in an agent interaction, and they represent the bulk of the generation workload despite being less frequent.

![Token distributions](stats/token_distributions.png)

## Usage with AIPerf

This dataset can be converted to [AIPerf](https://github.com/ai-dynamo/aiperf)'s `mooncake_trace` format for benchmarking. See [sammshen/agentic-dataset](https://github.com/sammshen/agentic-dataset) for the converter script and full documentation.

```bash
# Convert to mooncake_trace format (pre_gap → delay in ms)
python convert_lmcache_to_mooncake.py --output trace.jsonl

# Recommended: concurrent sessions, sequential intra-session turns
aiperf profile \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --concurrency 20 \
  --request-timeout-seconds 3600 \
  --extra-inputs ignore_eos:true \
  --use-server-token-count
```

AIPerf guarantees that requests within the same `session_id` run strictly sequentially in all scheduling modes. `--concurrency N` keeps N sessions active with no inter-turn delay (max cache pressure). For realistic timing with tool-execution gaps, use `--fixed-schedule` which honors the `delay` field from `pre_gap`.

## Intended Use

This dataset is designed for:

- **KV cache system benchmarking**: Evaluate prefix-aware caching strategies (e.g., LMCache, Mooncake, vLLM prefix caching, SGLang RadixAttention) using realistic workloads with verified prefix structure.
- **LLM serving research**: Study context growth patterns, request timing distributions, and output length distributions in real agentic workloads.
- **Cache policy design**: Use the per-source breakdown to understand how different workload types create different caching opportunities (e.g., SWE-bench's large steadily growing contexts vs GAIA's rapid growth and plateau).

## Citation

If you use this dataset, please cite:

```bibtex
@misc{lmcache-agentic-dataset-2026,
  title={LMCache Agentic Dataset: Multi-Turn LLM Agent Sessions for KV Cache Benchmarking},
  author={LMCache},
  year={2026},
  url={https://huggingface.co/datasets/sammshen/lmcache-agentic-traces}
}
```

## License

CC-BY-4.0