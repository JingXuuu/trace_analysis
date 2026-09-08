"""Offline full-tree node and token reuse distributions for the web UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from .formats import source_files


def source_key(path: Path) -> str:
    members = source_files(path)
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in members]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:24]


def reuse_path(path: Path, output_dir: Path) -> Path:
    return output_dir / f"reuse-v2-{source_key(path)}.svg"


def collect_reuse(cache, sizes):
    """Count nodes and their stored segment tokens once, excluding root."""
    nodes, tokens = Counter(), Counter()
    stack = list(cache.root_node.children.values())
    while stack:
        node = stack.pop()
        nodes[node.hit_count] += 1
        tokens[node.hit_count] += sum(sizes[key] for key in node.key)
        stack.extend(node.children.values())
    return nodes, tokens


def group_reuse(nodes, tokens):
    """Power-of-two ranges, including empty intermediate bins."""
    labels, node_counts, token_counts = [], [], []
    lower = 1
    while lower <= max(nodes, default=1):
        upper = lower * 2 - 1
        labels.append(str(lower) if lower == upper else f'{lower}–{upper}')
        node_counts.append(sum(count for hit, count in nodes.items() if lower <= hit <= upper))
        token_counts.append(sum(count for hit, count in tokens.items() if lower <= hit <= upper))
        lower *= 2
    return labels, node_counts, token_counts


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
                      build_sglang_radix_cache)
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
            token_distribution = {int(k): v for k, v in saved['tokens_by_hit_count'].items()}
            trace_format = saved.get('trace_format', 'lmcache_messages' if saved['estimated_tokens'] else 'native')
        else:
            print(f"Loading: {item['name']}", flush=True)
            (rows, sizes, _), trace_format = load_trace(path, 512)
            print(f"Building full tree: {len(rows):,} requests", flush=True)
            cache = build_sglang_radix_cache(rows)
            distribution, token_distribution = collect_reuse(cache, sizes)
            del rows, sizes, cache
            gc.collect()
        labels, counts, token_counts = group_reuse(distribution, token_distribution)
        total_nodes, total_tokens = sum(counts), sum(token_counts)
        shared_nodes = total_nodes - distribution.get(1, 0)
        shared_tokens = total_tokens - token_distribution.get(1, 0)
        estimated = trace_format == 'lmcache_messages'
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
            writer.writerow(['hit_count_range', 'node_count', 'stored_tokens'])
            writer.writerows(zip(labels, counts, token_counts))
        svg.with_suffix('.json').write_text(json.dumps({
            'total_nodes': total_nodes, 'total_tokens': total_tokens,
            'shared_nodes': shared_nodes, 'shared_tokens': shared_tokens,
            'estimated_tokens': estimated, 'root_excluded': True,
            'trace_format': trace_format,
            'nodes_by_hit_count': distribution, 'tokens_by_hit_count': token_distribution,
        }, indent=2) + '\n')
        temporary = svg.with_suffix('.tmp.svg')
        fig.savefig(temporary, format='svg')
        temporary.replace(svg)
        plt.close(fig)
        print(f"Saved: {svg}", flush=True)


if __name__ == '__main__':
    main()
