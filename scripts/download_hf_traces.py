#!/usr/bin/env python3
"""Download the external trace datasets used by this repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASETS = {
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
    parser.add_argument("--trace-root", type=Path, default=Path("traces/external"))
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
        print(f"Downloading {specification['repo_id']} -> {destination}", flush=True)
        for filename in specification["files"]:
            print(f"  {filename}", flush=True)
            hf_hub_download(
                repo_id=specification["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=specification["revision"],
                local_dir=destination,
            )
    print("Trace downloads complete.", flush=True)


if __name__ == "__main__":
    main()
