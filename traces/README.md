# Trace sources

## Mooncake

The `mooncake_trace*.jsonl` files contain normalized request paths with global
block hash IDs and 512-token blocks. They can be analyzed directly.

## External datasets

### Bailian and RAGPulse

```bash
python scripts/download_hf_traces.py bailian ragpulse
```

`bailian/` contains four production-derived Qwen workloads from
[Alibaba](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon), using
native 16-token blocks (Apache-2.0). A partial final block retains its used
length; IDs with different prefixes or used lengths are not merged.

`ragpulse/` contains the official [RAGPulse](https://github.com/flashserve/RAGPulse)
GitHub release (MIT), including `0_trace.jsonl` and all five content-length
lookups. This is the newer upstream release, not the older Hugging Face snapshot.
All six files are required. Only `0_trace.jsonl` appears in the trace selector.
Component order follows the upstream replay: JSON object order, then list order.
Token sizes are sums of published component lengths, not re-tokenized prompts.

The downloader pins both revisions and writes per-file hashes to `DOWNLOAD.json`.
Run `./scripts/prepare_reuse.sh` followed by `./scripts/export_site.sh` to publish
their derived charts without publishing the raw datasets.

### LMCache and CC

`lmcache-agentic-traces/` mirrors the data files and dataset card from
[zeelHz/lmcache-agentic-traces](https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces).
Its five Parquet shards contain cumulative message histories. The included
`lmcache_messages` adapter combines them into one trace in the web UI, using
message prefixes and explicitly estimated token sizes.

`cc-traces-weka-no-subagents-051226/` mirrors `traces.jsonl` and the
dataset card from
[semianalysisai/cc-traces-weka-no-subagents-051226](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226). It already
contains 64-token KV block hashes. Its `hash_id_scope` is `local`, so hash IDs
must be namespaced by top-level trace before sessions are combined.

From the repository root, install the download dependency and fetch both datasets
(approximately 5.14 GB combined):

```bash
python -m pip install '.[download,parquet]'
python scripts/download_hf_traces.py
```

To download just one dataset:

```bash
python scripts/download_hf_traces.py lmcache-agentic-traces
python scripts/download_hf_traces.py cc-traces-weka-no-subagents-051226
```

The script pins the source revisions and automatically places the files here:

```text
traces/
├── lmcache-agentic-traces/data/train-00000-of-00005.parquet
│                           ... through train-00004-of-00005.parquet
└── cc-traces-weka-no-subagents-051226/traces.jsonl
```

For manual downloads, use the links above and preserve this directory layout.
Refresh the web page after downloading to discover the new traces. For a custom
location, pass `--trace-root /path/to/traces` to the downloader and
`--trace-dir /path/to/traces` to `python -m trace_analysis.web`.

The upstream dataset cards included in each directory contain their licenses,
attribution, field definitions, and tokenizer caveats.

The multi-gigabyte data files are intentionally ignored by Git; dataset cards,
the downloader, and adapter code remain trackable. Running the downloader
restores the exact local data layout used by the web UI.
