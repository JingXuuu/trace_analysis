#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
web_python="${PYTHON:-$HOME/miniconda3/envs/sgl0514/bin/python3.12}"
if ! command -v "$web_python" >/dev/null 2>&1; then web_python=python3; fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/trace-analysis-${UID}/flashinfer}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/trace-analysis-${UID}/matplotlib}"
exec "$web_python" -m trace_analysis.reuse "$@"
