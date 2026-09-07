# Trace sources

## Mooncake

The `mooncake_trace*.jsonl` files contain normalized request paths with global
block hash IDs and 512-token blocks. They can be analyzed directly.

## External datasets

`external/lmcache-agentic-traces/` mirrors the data files and dataset card from
`zeelHz/lmcache-agentic-traces`. Its five Parquet shards contain cumulative
OpenAI-format message histories, not KV block hashes. It therefore needs a
message/token normalization adapter before it can be interpreted as a KV radix
tree.

`external/cc-traces-weka-no-subagents-051226/` mirrors `traces.jsonl` and the
dataset card from `semianalysisai/cc-traces-weka-no-subagents-051226`. It already
contains 64-token KV block hashes. Its `hash_id_scope` is `local`, so hash IDs
must be namespaced by top-level trace before sessions are combined.

Re-fetch or resume both datasets with:

```bash
/home/jingxu/miniconda3/envs/sgl0514/bin/python3.12 scripts/download_hf_traces.py
```

The upstream dataset cards included in each directory contain their licenses,
attribution, field definitions, and tokenizer caveats.

The multi-gigabyte data files are intentionally ignored by Git; dataset cards,
the downloader, and adapter code remain trackable. Running the downloader
restores the exact local data layout used by the web UI.
