"""Offline full-tree node reuse distributions for the web UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from .formats import source_files


def source_key(path: Path) -> str:
    members = source_files(path)
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in members]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:24]


def reuse_path(path: Path, output_dir: Path) -> Path:
    return output_dir / f"reuse-v2-{source_key(path)}.svg"


def group_reuse(nodes):
    """Power-of-two ranges, including empty intermediate bins."""
    labels, node_counts = [], []
    lower = 1
    while lower <= max(nodes, default=1):
        upper = lower * 2 - 1
        labels.append(str(lower) if lower == upper else f'{lower}–{upper}')
        node_counts.append(sum(count for hit, count in nodes.items() if lower <= hit <= upper))
        lower *= 2
    return labels, node_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, default=Path('traces'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/web'))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--replot', action='store_true', help='Redraw from saved counts without rebuilding trees')
    args = parser.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from .web import AnalysisService
    from .cli import (add_sglang_to_path, discover_sglang_python_root,
                      build_sglang_radix_cache, collect_hit_count_distribution)
    from .formats import load_trace
    import matplotlib.pyplot as plt
    import gc

    add_sglang_to_path(discover_sglang_python_root(None))
    service = AnalysisService([args.trace_dir], args.output_dir)
    for item in sorted(service.traces(), key=lambda item: item['bytes']):
        path = service.resolve_trace(item['id'])
        svg = reuse_path(path, service.output_dir)
        if svg.exists() and not args.force and not args.replot:
            print(f"Already prepared: {item['name']}", flush=True)
            continue
        if args.replot:
            saved = json.loads(svg.with_suffix('.json').read_text())
            distribution = {int(k): v for k, v in saved['nodes_by_hit_count'].items()}
        else:
            print(f"Loading: {item['name']}", flush=True)
            (rows, sizes, _), _ = load_trace(path, 512)
            print(f"Building full tree: {len(rows):,} requests", flush=True)
            cache = build_sglang_radix_cache(rows)
            distribution = collect_hit_count_distribution(cache)
            del rows, sizes, cache
            gc.collect()
        labels, counts = group_reuse(distribution)
        total_nodes = sum(counts)
        shared_nodes = total_nodes - distribution.get(1, 0)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.subplots_adjust(top=.68, bottom=.26, left=.09, right=.98)
        fig.set_facecolor('#fffaf4')
        ax.set_facecolor('#fffaf4')
        bars = ax.bar(range(len(labels)), counts, color=['#cbb196'] + ['#70845d'] * (len(labels)-1))
        ax.bar_label(bars, labels=[f'{value:,}' for value in counts], padding=4,
                     fontsize=9, rotation=90 if len(labels) > 12 else 0, color='#3d342d')
        ax.set_ylim(0, max(max(counts, default=0), 1) * 1.6)
        ax.set_xticks(range(len(labels)), labels, rotation=50, ha='right', fontsize=9)
        ax.set_xlabel('Hit-count range')
        ax.set_ylabel('Number of radix nodes')
        ax.grid(axis='y', alpha=.2, color='#8c5e3c')
        ax.set_axisbelow(True)
        fig.suptitle(f"{path.name} · Node reuse", y=.98, color='#3d342d', fontsize=16)
        summaries = [('Total nodes', f'{total_nodes:,}'),
                     ('Shared nodes', f'{shared_nodes:,}'),
                     ('Shared-node ratio', f'{shared_nodes / total_nodes:.1%}' if total_nodes else '0.0%')]
        for index, (label, value) in enumerate(summaries):
            fig.text(.2 + index * .3, .83, f'{label}\n{value}', ha='center', fontsize=13,
                     color='#3d342d', bbox=dict(boxstyle='round,pad=.5', facecolor='#f2e9df', edgecolor='#d9c9b8'))
        fig.text(.5, .02, 'Full tree · Root excluded · Shared = hit count > 1 · Sand: single-use; green: shared',
                 ha='center', fontsize=10, color='#786b60')
        fig.savefig(svg.with_suffix('.png'), dpi=160)
        import csv
        with svg.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['hit_count_range', 'node_count'])
            writer.writerows(zip(labels, counts))
        svg.with_suffix('.json').write_text(json.dumps({
            'total_nodes': total_nodes, 'shared_nodes': shared_nodes,
            'root_excluded': True, 'nodes_by_hit_count': distribution,
        }, indent=2) + '\n')
        temporary = svg.with_suffix('.tmp.svg')
        fig.savefig(temporary, format='svg')
        temporary.replace(svg)
        plt.close(fig)
        print(f"Saved: {svg}", flush=True)


if __name__ == '__main__':
    main()
