#!/usr/bin/env python3
"""Download the external trace datasets used by this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


DATASETS = {
    "bailian": {
        "repo_id": "alibaba-edu/qwen-bailian-usagetraces-anon",
        "revision": "5f7439c51ec248a0c585f7d90a41a6f57773b912",
        "source": "github-lfs",
        "files": ['README.md', 'LICENSE', *[f'qwen_{name}_blksz_16.jsonl'
                  for name in ('traceA', 'traceB', 'thinking', 'coder')]],
    },
    "ragpulse": {
        "repo_id": "flashserve/RAGPulse",
        "revision": "7da286becf0f049b2bcb1e5a11d9ba8eb638eff4",
        "source": "github",
        "files": ['README.md', 'LICENSE', *[f'data/{name}.jsonl' for name in
                  ('0_trace', '1_sys_prompt', '2_passages', '3_history', '4_user_input', '5_web_search')]],
    },
    "lmcache-agentic-traces": {
        "repo_id": "zeelHz/lmcache-agentic-traces",
        "revision": "565e5655d2e4b6cba0865a0e1d60b09ed2dfcdf0",
        "files": [
            "README.md",
            *[f"data/train-{index:05d}-of-00005.parquet" for index in range(5)],
        ],
    },
    "cc-traces-weka-no-subagents-051226": {
        "repo_id": "semianalysisai/cc-traces-weka-no-subagents-051226",
        "revision": "0ae681ae27a0e3e716b344cb21f1b01bb1313d52",
        "files": ["README.md", "traces.jsonl"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="datasets to fetch (default: all)",
    )
    parser.add_argument("--trace-root", type=Path, default=Path("traces"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.datasets or list(DATASETS)
    unknown = sorted(set(selected_datasets) - DATASETS.keys())
    if unknown:
        raise SystemExit(
            "unknown dataset(s): " + ", ".join(unknown)
            + "; available: " + ", ".join(DATASETS)
        )
    for name in selected_datasets:
        specification = DATASETS[name]
        destination = args.trace_root / name
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = []
        print(f"Downloading {specification['repo_id']} -> {destination}", flush=True)
        for filename in specification["files"]:
            print(f"  {filename}", flush=True)
            if specification.get('source', '').startswith('github'):
                target = destination / Path(filename).name
                host = ('media.githubusercontent.com/media' if specification['source'] == 'github-lfs'
                        and filename.endswith('.jsonl') else 'raw.githubusercontent.com')
                url = f"https://{host}/{specification['repo_id']}/{specification['revision']}/{filename}"
                temporary = target.with_suffix(target.suffix + '.part')
                if shutil.which('curl'):
                    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                    '--retry', '3', '--connect-timeout', '15', '--max-time', '600',
                                    '--output', str(temporary), url], check=True)
                else:
                    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                temporary.replace(target)
                digest = hashlib.sha256()
                with target.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                downloaded.append({'path': target.name, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest()})
                continue
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=specification["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=specification["revision"],
                local_dir=destination,
            )
        if downloaded:
            (destination / 'DOWNLOAD.json').write_text(json.dumps({
                'repo_id': specification['repo_id'], 'revision': specification['revision'],
                'files': downloaded}, indent=2) + '\n')
    print("Trace downloads complete.", flush=True)


if __name__ == "__main__":
    main()
