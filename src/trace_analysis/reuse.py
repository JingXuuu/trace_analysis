"""Offline full-tree node reuse and depth distributions for the web UI."""
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
                      build_sglang_radix_cache, collect_hit_count_distribution,
                      collect_depth_statistics)
    from .formats import load_trace
    import matplotlib.pyplot as plt
    import gc

    add_sglang_to_path(discover_sglang_python_root(None))
    service = AnalysisService([args.trace_dir], args.output_dir)
    for item in sorted(service.traces(), key=lambda item: item['bytes']):
        path = service.resolve_trace(item['id'])
        svg = reuse_path(path, service.output_dir)
        saved_path = svg.with_suffix('.json')
        saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
        if svg.exists() and 'nodes_by_depth' in saved and not args.force and not args.replot:
            print(f"Already prepared: {item['name']}", flush=True)
            continue
        if args.replot and 'nodes_by_depth' in saved:
            distribution = {int(k): v for k, v in saved['nodes_by_hit_count'].items()}
            depth_counts = {int(k): v for k, v in saved['nodes_by_depth'].items()}
        else:
            print(f"Loading: {item['name']}", flush=True)
            (rows, sizes, _), _ = load_trace(path, 512)
            print(f"Building full tree: {len(rows):,} requests", flush=True)
            cache = build_sglang_radix_cache(rows)
            distribution = collect_hit_count_distribution(cache)
            depth_counts = {depth: stats.total_nodes
                            for depth, stats in collect_depth_statistics(cache, len(rows)).items()
                            if depth > 0}
            del rows, sizes, cache
            gc.collect()
        labels, counts = group_reuse(distribution)
        total_nodes = sum(counts)
        shared_nodes = total_nodes - distribution.get(1, 0)
        fig, (ax, depth_ax) = plt.subplots(1, 2, figsize=(18, 5))
        fig.subplots_adjust(top=.68, bottom=.26, left=.055, right=.98, wspace=.25)
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
        fig.text(.27, .97, 'Node reuse', ha='center', va='top', color='#3d342d', fontsize=16)
        fig.text(.77, .97, 'Nodes by depth', ha='center', va='top', color='#3d342d', fontsize=16)
        summaries = [('Total nodes', f'{total_nodes:,}'),
                     ('Shared nodes', f'{shared_nodes:,}'),
                     ('Shared-node ratio', f'{shared_nodes / total_nodes:.1%}' if total_nodes else '0.0%')]
        for index, (label, value) in enumerate(summaries):
            fig.text(.12 + index * .15, .83, f'{label}\n{value}', ha='center', fontsize=12,
                     color='#3d342d', bbox=dict(boxstyle='round,pad=.5', facecolor='#f2e9df', edgecolor='#d9c9b8'))
        depths = sorted(depth_counts)
        values = [depth_counts[d] for d in depths]
        depth_ax.set_facecolor('#fffaf4')
        depth_ax.plot(depths, values, color='#70845d', linewidth=2,
                      marker='o', markersize=3)
        depth_ax.fill_between(depths, values, color='#70845d', alpha=.12)
        depth_ax.set_xlabel('Compressed radix depth')
        depth_ax.set_ylabel('Number of radix nodes')
        depth_ax.set_ylim(bottom=0)
        from matplotlib.ticker import MaxNLocator
        depth_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        depth_ax.grid(axis='y', alpha=.2, color='#8c5e3c')
        depth_ax.set_axisbelow(True)
        peak_depth = max(depths, key=depth_counts.get) if depths else 0
        depth_summaries = [('Max depth', f'{max(depths, default=0):,}'),
                           ('Widest depth', str(peak_depth)),
                           ('Peak nodes', f'{depth_counts.get(peak_depth, 0):,}')]
        for index, (label, value) in enumerate(depth_summaries):
            fig.text(.62 + index * .15, .83, f'{label}\n{value}', ha='center', fontsize=12,
                     color='#3d342d', bbox=dict(boxstyle='round,pad=.5', facecolor='#f2e9df', edgecolor='#d9c9b8'))
        from matplotlib.lines import Line2D
        fig.add_artist(Line2D([.515, .515], [.08, .96], transform=fig.transFigure,
                              color='#d9c9b8', linewidth=1))
        fig.text(.5, .02, 'Full tree · Root excluded · Independent of display limits · Shared = hit count > 1',
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
            'nodes_by_depth': depth_counts,
        }, indent=2) + '\n')
        temporary = svg.with_suffix('.tmp.svg')
        fig.savefig(temporary, format='svg')
        # Embed exact counts and plot geometry for offline browser interaction.
        import xml.etree.ElementTree as ET
        document = ET.parse(temporary)
        metadata = ET.SubElement(document.getroot(), '{http://www.w3.org/2000/svg}metadata',
                                 {'id': 'depth-profile-data'})
        bounds = depth_ax.get_position()
        metadata.text = json.dumps({'counts': depth_counts,
                                    'bounds': [bounds.x0, 1 - bounds.y1, bounds.width, bounds.height],
                                    'xlim': list(depth_ax.get_xlim()),
                                    'ylim': list(depth_ax.get_ylim())})
        document.write(temporary, encoding='utf-8', xml_declaration=True)
        temporary.replace(svg)
        plt.close(fig)
        print(f"Saved: {svg}", flush=True)


if __name__ == '__main__':
    main()
