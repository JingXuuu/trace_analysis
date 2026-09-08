# Trace Radix Analysis

This repository turns a trace into the same compressed radix-tree
data structure used by SGLang's prefix cache, validates the result against every
request path, and exports both the complete structure and readable summaries.

The analyzer uses
`sglang.srt.mem_cache.radix_cache.RadixCache.create_simulated(page_size=1)`.
Each Mooncake `hash_id` is inserted as one radix-key unit; the exact token size
(512 tokens for a complete block and the actual size of a final partial block)
is tracked separately.

## Included data

- `traces/mooncake_trace.jsonl`: all 23,608 requests.
- `traces/mooncake_trace_1000.jsonl`: a 1,000-request sample.
- `traces/mooncake_trace_short.jsonl`: a ten-request smoke-test trace.
- [`traces/cc-traces-weka-no-subagents-051226/traces.jsonl`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226): 949
  SemiAnalysis traces containing 136,118 requests and native 64-token block
  hashes.
- [`traces/lmcache-agentic-traces/data/`](https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces): five message-level Parquet
  shards containing 24,880 requests.
- `docs/`: current pre-rendered trees and reuse charts for the static website.
- `traces/bailian/`: Qwen/Bailian chat, API, reasoning, and coding workloads
  with fixed 16-token block IDs.
- `traces/ragpulse/`: RAGPulse requests and five variable-length content-ID
  lookup files from the pinned official GitHub release.

The external data can be restored or updated reproducibly with
`make download-traces` after installing the download dependencies described in
[DEPLOYMENT.md](DEPLOYMENT.md).
Repository revisions and expected SHA-256 hashes are recorded in
`traces/SOURCES.json`. The multi-gigabyte data files remain ignored by
Git.

Download just the new datasets with `python scripts/download_hf_traces.py bailian ragpulse`.
Their pinned GitHub revisions are in the downloader; `DOWNLOAD.json` records
downloaded file sizes and SHA-256 checksums. Raw JSONL files stay ignored by Git.
Bailian files are separate traces, with IDs compared only within a file.
RAGPulse preserves the upstream replay's component order and uses supplied chunk
token lengths. Repeated content after a different prefix becomes a different
radix edge; this is prefix reuse, not arbitrary document-cache reuse. The tree
does not infer token-level overlap inside different content chunks.
RAGPulse component lengths differ slightly from reported full input lengths;
node sizes represent the supplied component totals, without inventing missing tokens.

## Run it

Block sizes for the included traces are fixed by their source data:

| Trace | Block / radix-unit size |
| --- | --- |
| Mooncake (all three files) | 512 tokens per block |
| CC | 64 tokens per block |
| Bailian (all four workloads) | 16 tokens per block |
| RAGPulse | Variable-length chunks; exact lengths supplied per ID |
| LMCache | One message per unit; token lengths estimated |

Final blocks may be shorter. A compressed radix node can contain multiple units,
so its displayed token size is not the source block size.

Set up Python, SGLang, and Graphviz as described in
[DEPLOYMENT.md](DEPLOYMENT.md), then analyze the included Mooncake trace:

```bash
make analyze PYTHON=python3
```

For another trace or checkout:

```bash
PYTHONPATH=src SGLANG_PYTHON_ROOT=/path/to/sglang/python \
  python -m trace_analysis /path/to/trace.jsonl \
  --output-dir artifacts/my_trace \
  --max-nodes 1000 \
  --max-nodes-per-depth 16 \
  --priority depth
```

The SGLang source path can also be passed with `--sglang-python-root`. When it
is omitted, the tool checks `SGLANG_PYTHON_ROOT` and then looks for a sibling
`sglang/python` checkout. Use `--no-render` if Graphviz or matplotlib is not
available.

Install the command into an existing environment with `python -m pip install -e
.`; this provides the equivalent `trace-radix` executable. SGLang itself is an
external source dependency because this tool intentionally tests the exact
checkout selected by the user.

Mooncake JSONL, SemiAnalysis CC JSONL, and LMCache message Parquet are registered
input adapters. The CC adapter reads its embedded 64-token block size and
namespaces hash IDs by source trace and model because the upstream IDs are
explicitly local. The LMCache data contains cumulative messages rather than KV
block hashes, so its adapter creates deterministic, model-scoped message-prefix
units. Their displayed token sizes are clearly marked as estimates based on
canonical UTF-8 byte length; they are not presented as exact KV block sizes.

The tree builder and renderers consume normalized request paths, so another
format can be added by registering a `TraceAdapter` in
`src/trace_analysis/formats.py`. Both the CLI and web UI support `auto` format
detection.

## Interactive web plotter

After installing the dependencies, start the web plotter:

```bash
PYTHON=python3 ./scripts/run_web.sh --port 9000
```

Open `http://<server-ip>:9000` (or `http://localhost:9000` on the server).
Stop with Ctrl+C. Without `--port`, the default is 8080. The launcher works from
any working directory, with relative trace/output paths resolved from the repo.
It does not install dependencies. On a new machine, follow [DEPLOYMENT.md](DEPLOYMENT.md).
Override the Python environment when needed:

```bash
PYTHON=/path/to/python SGLANG_PYTHON_ROOT=/path/to/sglang/python \
  ./scripts/run_web.sh --port 9000
```

Use `./scripts/run_web.sh --help` for options including `--host`, `--trace-dir`,
and `--output-dir`.

The page lets you select any supported trace
under `traces/`, edit the whole-tree and per-depth limits, view the trace's fixed unit size,
and plot with depth-first priority. It renders in the page
and provides direct SVG and PNG downloads. Hovering a displayed node highlights
that node, all of its ancestors, all of its descendants, and the connecting
edges while dimming unrelated branches. Add more input directories by running
`python -m trace_analysis.web --trace-dir /path/one --trace-dir /path/two`.

The `lmcache-agentic-traces` dataset appears as one trace: all its Parquet shards
are combined with shared prefix accounting across shards.

The server caches the last complete SGLang tree, so changing only visualization
settings avoids reparsing the trace. It has no frontend CDN dependency. The
default server bind is `127.0.0.1`; the Make target uses `0.0.0.0` so it can be
opened remotely when the host firewall permits it.

## Outputs

Prepare full-tree reuse histograms offline for every available trace:

```bash
./scripts/prepare_reuse.sh
```

The selected trace's node-reuse chart appears above the tree after refreshing the
page, without rebuilding the tree. Hit counts are grouped into 1, 2–3, 4–7,
8–15, and successive doubling ranges. Bars show exact node counts. Summary
cards show total nodes, shared nodes, and the shared-node percentage.
Shared means hit count > 1; the root is excluded. LMCache token sizes are
explicitly estimated. All LMCache
shards count as one trace. SVG, PNG, and CSV files are saved in `artifacts/web`.
Use `--output-dir` to match a custom web output directory. Existing plots are
reused; use `--force` to regenerate. Changed source files get a new cache key.
Use `--replot` to redraw charts from saved counts without rebuilding the trees.
Local `artifacts/` outputs are ignored by Git; only the current static site in
`docs/` is published.
Full CC and LMCache preparation can take several minutes and substantial RAM.

Every run writes:

- `*_radix_tree_full.json`: the complete parent-linked radix tree, excluding
  hash values from node labels and records.
- `*_radix_tree.svg` and `*.png`: the tree view, limited by `--max-nodes`
  (root included) and `--max-nodes-per-depth`.
- `*_hit_count_distribution.csv`: exact `(hit_count, node_count)` frequencies
  for the full tree.
- `*_hit_count_distribution.svg` and `*.png`: linear low-count detail plus the
  full log-log distribution.
- `*_manifest.json`: input checksum, validation counts, limits, and output paths.

Each displayed radix depth is enclosed by a dashed column. The column header
shows nearest-rank p50 and p99 `hit_count`, full-tree node count, parent-node
count, sink-node count, and the number retained in the displayed graph.

At every displayed depth, the graph shares the node budget evenly among the
retained parents using round-robin allocation. Siblings are ranked by
`TreeNode.hit_count`; if a parent has fewer children than its share, its unused
slots are redistributed. Thus, with 16 slots and four parents, each parent gets
up to four children while every descendant stays connected to its true parent.

With `--priority depth`, shallower levels consume the whole-tree limit first.
With `--priority hit_count`, selection follows the highest-hit connected
frontier from the fairly constructed per-depth candidate tree.

Node colors mean:

- Blue: any node used by more than one request (`hit_count > 1`).
- Green: a leaf node used by exactly one request.
- Gray: an internal node used by exactly one request.

Every node label shows only its compressed segment's token size and
`hit_count`; hash IDs are deliberately omitted.

## Test

```bash
make test
```

The tests cover partial-block sizing, local-hash namespacing, depth statistics,
graph limits, parent-preserving selection, correct parent edges, complete-tree
parent links, color semantics, and hit-count accounting.

## Deploy

For GitHub Pages, pre-render the default trees with `./scripts/export_site.sh`.
The generated `docs/` site supports trace selection, hover highlighting, reuse
charts, and downloads without a backend. See the GitHub Pages section in
[DEPLOYMENT.md](DEPLOYMENT.md) for setup. Graph settings remain editable in the
Python web server; they are fixed in the static preview.

The repository includes a Dockerfile, Compose configuration, health endpoint,
environment template, ignored/reproducible external data, and GitHub Actions
CI. See [DEPLOYMENT.md](DEPLOYMENT.md) for clone, local-environment, and
container deployment instructions.
