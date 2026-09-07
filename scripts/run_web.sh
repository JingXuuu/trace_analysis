#!/usr/bin/env bash
# Start the web UI from any working directory; extra arguments go to the server.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${PYTHON:-}" ]]; then
    web_python="$PYTHON"
else
    web_python=""
    for candidate in "$repo_root/.venv/bin/python" \
        "${CONDA_PREFIX:-/nonexistent}/bin/python" \
        "$HOME/miniconda3/envs/sgl0514/bin/python3.12" python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("torch") is None)' 2>/dev/null; then
            web_python="$candidate"
            break
        fi
    done
    web_python="${web_python:-python3}"
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${TMPDIR:-/tmp}/trace-analysis-${UID}/flashinfer}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/trace-analysis-${UID}/matplotlib}"
if [[ -z "${SGLANG_PYTHON_ROOT:-}" && -f "$repo_root/../sglang/python/sglang/srt/mem_cache/radix_cache.py" ]]; then
    export SGLANG_PYTHON_ROOT="$repo_root/../sglang/python"
fi

# argparse handles --help and validates options. Later arguments override defaults.
exec "$web_python" -m trace_analysis.web \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" "$@"
