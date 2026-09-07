# Deployment

## Git repository contents

Source code, tests, configuration, dataset cards, and dataset checksums are
tracked normally. Raw external Parquet and multi-gigabyte JSONL files are
excluded by `.gitignore`; restore them after cloning with:

```bash
python -m pip install '.[download]'
python scripts/download_hf_traces.py
```

The downloader pins the exact Hugging Face revisions recorded in
`traces/external/SOURCES.json` and resumes existing downloads.

## Publish to a Git host

The repository is self-contained apart from the ignored external trace blobs.
To publish a fresh checkout:

```bash
git init -b main
git add .
git commit -m "Initial trace radix analyzer"
git remote add origin <repository-url>
git push -u origin main
```

Do not force-add the ignored Parquet or multi-gigabyte JSONL files. Their pinned
source revisions, sizes, and SHA-256 checksums are versioned instead, and the
download script restores them after a clone.

## Existing SGLang checkout

For a lightweight deployment using an existing SGLang environment:

```bash
python -m pip install -e '.[parquet]'
SGLANG_PYTHON_ROOT=/path/to/sglang/python trace-radix-web \
  --host 0.0.0.0 --port 8080
```

The server also discovers an installed `sglang` package or a sibling
`sglang/python` checkout automatically.

## Container

The image installs Graphviz, SGLang 0.5.14, PyArrow, and this package. External
trace data is mounted at runtime instead of copied into the image.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8080`. Change `TRACE_ANALYSIS_PORT` in `.env` when the
default port is occupied. Before building, set `TRACE_ANALYSIS_UID` and
`TRACE_ANALYSIS_GID` to the owner of the local `artifacts/` directory if they
are not 1000. These values are applied while building the non-root container
user, including ownership of its writable cache and output directories.

Compose mounts the clone's `traces/` directory read-only and exposes it as the
single trace source, so built-in inputs are not duplicated in the selector.

The health check uses `GET /api/health`. Generated SVG and PNG files are written
to the bind-mounted `artifacts/` directory.

## Continuous integration

`.github/workflows/ci.yml` installs the core and Parquet dependencies, runs the
unit tests, and compiles all Python entry points on every push and pull request.
