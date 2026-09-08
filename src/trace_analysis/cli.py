#!/usr/bin/env python3
"""Build and render a Mooncake trace with SGLang's simulated RadixCache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from array import array
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .formats import (
    TraceRow,
    available_formats,
    load_trace as load_trace_with_format,
    token_size_semantics,
)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A selected view of an SGLang TreeNode."""

    graph_id: str
    parent_graph_id: str
    token_size: int
    hit_count: int
    is_leaf: bool
    depth: int


@dataclass(frozen=True, slots=True)
class DepthStats:
    """Full-tree statistics for one compressed radix depth."""

    depth: int
    p50_hit_count: int
    p99_hit_count: int
    max_hit_count: int
    total_nodes: int
    parent_nodes: int
    sink_nodes: int


def discover_sglang_python_root(explicit_root: Path | None) -> Path:
    """Locate SGLang from the CLI, environment, or a sibling checkout."""
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    if environment_root := os.environ.get("SGLANG_PYTHON_ROOT"):
        candidates.append(Path(environment_root))

    repository_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repository_root.parent / "sglang" / "python",
            Path.cwd().parent / "sglang" / "python",
        ]
    )
    for candidate in candidates:
        source_file = candidate / "sglang/srt/mem_cache/radix_cache.py"
        if source_file.is_file():
            return candidate.resolve()

    installed_spec = importlib.util.find_spec("sglang")
    if installed_spec and installed_spec.submodule_search_locations:
        package_root = Path(next(iter(installed_spec.submodule_search_locations)))
        installed_root = package_root.parent
        if (package_root / "srt/mem_cache/radix_cache.py").is_file():
            return installed_root.resolve()

    searched = "\n  - ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "Could not find an SGLang source checkout. Pass --sglang-python-root "
        "or set SGLANG_PYTHON_ROOT. Searched:\n  - " + searched
    )


def add_sglang_to_path(sglang_python_root: Path) -> None:
    """Make the selected SGLang package importable."""
    root = str(sglang_python_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_trace(
    trace_file: Path, block_size: int, trace_format: str = "auto"
) -> tuple[list[TraceRow], dict[int, int], int]:
    """Compatibility wrapper returning normalized rows from an input adapter."""
    result, _ = load_trace_with_format(trace_file, block_size, trace_format)
    return result


def build_sglang_radix_cache(rows: list[TraceRow]) -> Any:
    """Insert every trace path into SGLang's production radix data structure."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    cache = RadixCache.create_simulated(page_size=1)
    for row in rows:
        key = RadixKey(token_ids=array("q", row.hash_ids))
        cache.insert(InsertParams(key=key))
    return cache


def node_token_size(node: Any, block_sizes: dict[int, int]) -> int:
    """Sum exact Mooncake token sizes for an SGLang compressed node key."""
    return sum(block_sizes[hash_id] for hash_id in node.key)


def _nearest_rank(values: list[int], percentile: float) -> int:
    """Calculate a deterministic nearest-rank percentile without NumPy."""
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def collect_depth_statistics(cache: Any, total_requests: int) -> dict[int, DepthStats]:
    """Collect hit-count and topology statistics for every full-tree depth."""
    buckets: dict[int, list[Any]] = defaultdict(list)
    buckets[0].append(cache.root_node)
    queue = deque((child, 1) for child in cache.root_node.children.values())
    while queue:
        node, depth = queue.popleft()
        buckets[depth].append(node)
        queue.extend((child, depth + 1) for child in node.children.values())

    result: dict[int, DepthStats] = {}
    for depth, depth_nodes in sorted(buckets.items()):
        hit_counts = (
            [total_requests]
            if depth == 0
            else [node.hit_count for node in depth_nodes]
        )
        parent_nodes = sum(bool(node.children) for node in depth_nodes)
        result[depth] = DepthStats(
            depth=depth,
            p50_hit_count=_nearest_rank(hit_counts, 0.50),
            p99_hit_count=_nearest_rank(hit_counts, 0.99),
            max_hit_count=max(hit_counts),
            total_nodes=len(depth_nodes),
            parent_nodes=parent_nodes,
            sink_nodes=len(depth_nodes) - parent_nodes,
        )
    return result


def validate_sglang_tree(
    cache: Any,
    rows: list[TraceRow],
    block_sizes: dict[int, int],
) -> dict[str, int]:
    """Validate parent links, child indexes, usage counts, and every trace path."""
    root = cache.root_node
    nodes: list[Any] = []
    seen_block_ids: set[int] = set()
    stack = [root]

    while stack:
        node = stack.pop()
        for child_key, child in node.children.items():
            if child.parent is not node:
                raise ValueError(f"SGLang TreeNode {child.id} has an invalid parent")
            if child.key.child_key(cache.page_size) != child_key:
                raise ValueError(f"SGLang TreeNode {child.id} has an invalid child key")
            for hash_id in child.key:
                if hash_id in seen_block_ids:
                    raise ValueError(f"hash_id {hash_id} occurs in multiple tree nodes")
                seen_block_ids.add(hash_id)
            nodes.append(child)
            stack.append(child)

    calculated_hits = {id(node): 0 for node in nodes}
    for row in rows:
        node = root
        offset = 0
        while offset < len(row.hash_ids):
            child_key = row.hash_ids[offset]
            child = node.children.get(child_key)
            if child is None:
                raise ValueError(
                    f"line {row.line_number}: path is absent at hash offset {offset}"
                )
            segment = tuple(child.key)
            if row.hash_ids[offset : offset + len(segment)] != segment:
                raise ValueError(
                    f"line {row.line_number}: path diverges at hash offset {offset}"
                )
            calculated_hits[id(child)] += 1
            offset += len(segment)
            node = child

    for node in nodes:
        if node.hit_count != calculated_hits[id(node)]:
            raise ValueError(
                f"SGLang TreeNode {node.id} hit_count={node.hit_count}, "
                f"expected {calculated_hits[id(node)]}"
            )

    if seen_block_ids != block_sizes.keys():
        raise ValueError("SGLang tree block IDs do not match the source trace")

    return {
        "validated_requests": len(rows),
        "validated_sglang_nodes_excluding_root": len(nodes),
        "validated_unique_cache_blocks": len(seen_block_ids),
        "validated_unique_cache_tokens": sum(
            node_token_size(node, block_sizes) for node in nodes
        ),
    }


def select_graph_nodes(
    cache: Any,
    block_sizes: dict[int, int],
    max_nodes: int,
    max_nodes_per_depth: int,
    priority: str = "depth",
) -> tuple[list[GraphNode], dict[int, int]]:
    """Select a connected tree view using depth or hit-count priority."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    if max_nodes_per_depth < 1:
        raise ValueError("max_nodes_per_depth must be at least 1")
    if priority not in {"depth", "hit_count"}:
        raise ValueError("priority must be 'depth' or 'hit_count'")

    # First construct a parent-connected candidate tree. Every depth shares its
    # budget round-robin among the retained parents at the preceding depth.
    candidates: list[GraphNode] = []
    current_depth = [(child, "ROOT") for child in cache.root_node.children.values()]
    depth = 1

    while current_depth:
        children_by_parent: dict[str, list[tuple[Any, str]]] = {}
        for candidate in current_depth:
            children_by_parent.setdefault(candidate[1], []).append(candidate)

        for children in children_by_parent.values():
            children.sort(key=lambda item: (-item[0].hit_count, item[0].id))

        # Give every retained parent an equal number of turns. Parents are
        # ordered by their most-used remaining child, so only the remainder
        # slots favor higher usage. If a parent runs out of children, its unused
        # share is naturally redistributed in later rounds.
        depth_budget = max_nodes_per_depth
        if priority == "depth":
            depth_budget = min(depth_budget, max_nodes - 1 - len(candidates))
            if depth_budget <= 0:
                break
        ranked: list[tuple[Any, str]] = []
        while len(ranked) < depth_budget:
            active_parents = sorted(
                (
                    (parent_id, children)
                    for parent_id, children in children_by_parent.items()
                    if children
                ),
                key=lambda item: (
                    -item[1][0][0].hit_count,
                    item[1][0][0].id,
                    item[0],
                ),
            )
            if not active_parents:
                break
            for _, children in active_parents:
                if len(ranked) >= depth_budget:
                    break
                ranked.append(children.pop(0))
        next_depth: list[tuple[Any, str]] = []

        for node, parent_graph_id in ranked:
            graph_id = f"N{len(candidates)}"
            candidates.append(
                GraphNode(
                    graph_id=graph_id,
                    parent_graph_id=parent_graph_id,
                    token_size=node_token_size(node, block_sizes),
                    hit_count=node.hit_count,
                    is_leaf=not node.children,
                    depth=depth,
                )
            )
            next_depth.extend((child, graph_id) for child in node.children.values())

        current_depth = next_depth
        depth += 1

    if priority == "depth":
        selected = candidates
    else:
        children_by_parent_id: dict[str, list[GraphNode]] = {}
        for node in candidates:
            children_by_parent_id.setdefault(node.parent_graph_id, []).append(node)

        frontier = [
            (-node.hit_count, node.depth, node.graph_id, node)
            for node in children_by_parent_id.get("ROOT", [])
        ]
        heapq.heapify(frontier)
        selected = []
        while frontier and len(selected) + 1 < max_nodes:
            _, _, _, node = heapq.heappop(frontier)
            selected.append(node)
            for child in children_by_parent_id.get(node.graph_id, []):
                heapq.heappush(
                    frontier,
                    (-child.hit_count, child.depth, child.graph_id, child),
                )

        # Reassign compact graph IDs after best-first filtering. A node is only
        # unlocked after its parent is selected, so the parent map always exists.
        graph_id_map = {"ROOT": "ROOT"}
        compact_selected: list[GraphNode] = []
        for index, node in enumerate(selected):
            compact_id = f"N{index}"
            compact_selected.append(
                GraphNode(
                    graph_id=compact_id,
                    parent_graph_id=graph_id_map[node.parent_graph_id],
                    token_size=node.token_size,
                    hit_count=node.hit_count,
                    is_leaf=node.is_leaf,
                    depth=node.depth,
                )
            )
            graph_id_map[node.graph_id] = compact_id
        selected = compact_selected

    displayed_by_depth = dict(sorted(Counter(node.depth for node in selected).items()))
    return selected, displayed_by_depth


def dot_text(
    nodes: list[GraphNode],
    total_requests: int,
    depth_statistics: dict[int, DepthStats] | None = None,
    token_label: str = "tokens",
) -> str:
    """Create a Graphviz tree with a dashed, annotated column per depth."""
    displayed_by_depth: dict[int, list[GraphNode]] = defaultdict(list)
    for node in nodes:
        displayed_by_depth[node.depth].append(node)

    if depth_statistics is None:
        depth_statistics = {
            0: DepthStats(0, total_requests, total_requests, total_requests, 1, 1, 0)
        }
        for depth, depth_nodes in displayed_by_depth.items():
            hits = [node.hit_count for node in depth_nodes]
            parents = sum(not node.is_leaf for node in depth_nodes)
            depth_statistics[depth] = DepthStats(
                depth=depth,
                p50_hit_count=_nearest_rank(hits, 0.50),
                p99_hit_count=_nearest_rank(hits, 0.99),
                max_hit_count=max(hits),
                total_nodes=len(depth_nodes),
                parent_nodes=parents,
                sink_nodes=len(depth_nodes) - parents,
            )

    def header_label(stats: DepthStats, displayed: int) -> str:
        return (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="1">'
            f'<TR><TD ALIGN="LEFT"><B>Depth {stats.depth}</B></TD></TR>'
            f'<TR><TD ALIGN="LEFT">p50 hit={stats.p50_hit_count:,}  |  '
            f'p99 hit={stats.p99_hit_count:,}  |  max hit={stats.max_hit_count:,}</TD></TR>'
            f'<TR><TD ALIGN="LEFT">total nodes={stats.total_nodes:,}  |  '
            f'shown={displayed:,}</TD></TR>'
            f'<TR><TD ALIGN="LEFT">parent nodes={stats.parent_nodes:,}  |  '
            f'sink nodes={stats.sink_nodes:,}</TD></TR>'
            '</TABLE>>'
        )

    lines = [
        "digraph RadixTree {",
        '  graph [rankdir=LR, bgcolor=white, pad=0.25, nodesep=0.18, ranksep=0.45, splines=polyline, compound=true, newrank=true];',
        '  node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10, margin="0.10,0.06"];',
        '  edge [color="#718096", penwidth=0.8, arrowsize=0.55];',
        "  subgraph cluster_depth_0 {",
        "    rank=same;",
        '    graph [label="", style=dashed, color="#94a3b8", penwidth=1.2, margin=12];',
        f'    HEADER_0 [shape=plain, style="", class="depth-header", group="depth_headers", fontname="DejaVu Sans", fontsize=10, fontcolor="#334155", label={header_label(depth_statistics[0], 1)}];',
        f'    ROOT [label="ROOT\\n0 {token_label}\\nrequests={total_requests:,}", fillcolor="#1a202c", color="#ffffff", fontcolor="#ffffff", penwidth=2.2];',
        '    HEADER_0 -> ROOT [style=invis, constraint=false, weight=100];',
        "  }",
    ]
    for depth, depth_nodes in sorted(displayed_by_depth.items()):
        stats = depth_statistics[depth]
        lines.extend(
            [
                f"  subgraph cluster_depth_{depth} {{",
                "    rank=same;",
                '    graph [label="", style=dashed, color="#94a3b8", penwidth=1.2, margin=12];',
                f'    HEADER_{depth} [shape=plain, style="", class="depth-header", group="depth_headers", fontname="DejaVu Sans", fontsize=10, fontcolor="#334155", label={header_label(stats, len(depth_nodes))}];',
            ]
        )
        for node in depth_nodes:
            if node.hit_count > 1:
                fill, stroke = "#2b6cb0", "#90cdf4"
            elif node.is_leaf:
                fill, stroke = "#2f855a", "#9ae6b4"
            else:
                fill, stroke = "#4a5568", "#a0aec0"
            lines.append(
                f'    {node.graph_id} [label="{node.token_size:,} {token_label}\\n'
                f'hit_count={node.hit_count:,}", '
                f'fillcolor="{fill}", color="{stroke}", fontcolor="#ffffff"];'
            )
        lines.append(
            f"    HEADER_{depth} -> {depth_nodes[0].graph_id} "
            "[style=invis, constraint=false, weight=100];"
        )
        lines.append("  }")
    header_ids = ["HEADER_0", *[f"HEADER_{depth}" for depth in sorted(displayed_by_depth)]]
    if len(header_ids) > 1:
        lines.append(
            "  " + " -> ".join(header_ids)
            + " [style=invis, weight=100, minlen=2];"
        )
    for node in nodes:
        lines.append(f"  {node.parent_graph_id} -> {node.graph_id};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_graph(dot: str, output_file: Path, output_format: str) -> None:
    """Render DOT text through the local Graphviz executable."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = ["dot", f"-T{output_format}", "-o", str(output_file)]
    if output_format == "png":
        command.insert(1, "-Gdpi=120")
    subprocess.run(command, input=dot.encode(), check=True)


def collect_hit_count_distribution(cache: Any) -> Counter[int]:
    """Count full-tree SGLang radix nodes by exact TreeNode.hit_count."""
    distribution: Counter[int] = Counter()
    stack = list(cache.root_node.children.values())
    while stack:
        node = stack.pop()
        distribution[node.hit_count] += 1
        stack.extend(node.children.values())
    return distribution


def full_tree_document(
    cache: Any,
    block_sizes: dict[int, int],
    total_requests: int,
    size_semantics: str = "native_trace_tokens",
) -> dict[str, Any]:
    """Return the complete radix-tree structure without exposing hash IDs."""
    records: list[dict[str, Any]] = [
        {
            "id": "ROOT",
            "parent_id": None,
            "depth": 0,
            "token_size": 0,
            "hit_count": total_requests,
            "is_leaf": not cache.root_node.children,
            "child_count": len(cache.root_node.children),
        }
    ]
    queue = deque(
        (child, "ROOT", 1)
        for child in sorted(cache.root_node.children.values(), key=lambda node: node.id)
    )
    next_id = 0
    while queue:
        node, parent_id, depth = queue.popleft()
        node_id = f"N{next_id}"
        next_id += 1
        records.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "depth": depth,
                "token_size": node_token_size(node, block_sizes),
                "hit_count": node.hit_count,
                "is_leaf": not node.children,
                "child_count": len(node.children),
            }
        )
        queue.extend(
            (child, node_id, depth + 1)
            for child in sorted(node.children.values(), key=lambda child: child.id)
        )
    return {
        "schema": "trace-analysis.radix-tree.v1",
        "description": (
            "Complete SGLang radix tree. Each non-root record is one compressed "
            "radix node; source hash IDs are intentionally omitted."
        ),
        "token_size_semantics": size_semantics,
        "nodes": records,
    }


def write_json(document: Any, output_file: Path) -> None:
    """Write deterministic, human-readable JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_hit_count_distribution_csv(
    distribution: Counter[int], output_file: Path
) -> None:
    """Write exact hit-count frequencies for downstream analysis."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hit_count", "node_count"])
        writer.writerows(sorted(distribution.items()))


def plot_hit_count_distribution(
    distribution: Counter[int], output_files: list[Path]
) -> None:
    """Plot low-hit detail and the complete long-tailed distribution."""
    import matplotlib.pyplot as plt

    hit_counts = sorted(distribution)
    node_counts = [distribution[hit_count] for hit_count in hit_counts]
    detail_max = min(20, hit_counts[-1])
    detail_hits = list(range(1, detail_max + 1))
    detail_counts = [distribution.get(hit_count, 0) for hit_count in detail_hits]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (detail_axis, full_axis) = plt.subplots(
        1, 2, figsize=(14, 6.5), dpi=180
    )
    figure.suptitle(
        "SGLang Radix-Tree Node Hit-Count Distribution",
        fontsize=16,
        fontweight="bold",
    )

    detail_axis.bar(detail_hits, detail_counts, color="#2b6cb0", width=0.82)
    detail_axis.set_title("Low hit-count detail")
    detail_axis.set_xlabel("TreeNode.hit_count")
    detail_axis.set_ylabel("Number of radix nodes")
    detail_axis.set_xticks(detail_hits)
    detail_axis.tick_params(axis="x", labelrotation=45)

    full_axis.plot(
        hit_counts,
        node_counts,
        color="#2b6cb0",
        linewidth=1,
        alpha=0.65,
    )
    full_axis.scatter(
        hit_counts,
        node_counts,
        color="#2f855a",
        edgecolor="white",
        linewidth=0.4,
        s=24,
        zorder=3,
    )
    full_axis.set_xscale("log")
    full_axis.set_yscale("log")
    full_axis.set_title("Full range (log-log)")
    full_axis.set_xlabel("TreeNode.hit_count")
    full_axis.set_ylabel("Number of radix nodes")
    full_axis.grid(True, which="both", alpha=0.25)

    total_nodes = sum(node_counts)
    reused_nodes = total_nodes - distribution.get(1, 0)
    figure.text(
        0.5,
        0.01,
        (
            f"Full SGLang tree, excluding root: {total_nodes:,} nodes; "
            f"hit_count > 1: {reused_nodes:,} nodes; max hit_count: {hit_counts[-1]:,}"
        ),
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))

    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, export, and render a Mooncake trace with SGLang's "
            "simulated RadixCache."
        )
    )
    parser.add_argument("trace_file", type=Path)
    parser.add_argument(
        "--trace-format",
        choices=("auto", *available_formats()),
        default="auto",
        help="input adapter (default: auto)",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-nodes-per-depth", type=int, default=16)
    parser.add_argument(
        "--priority",
        choices=("depth", "hit_count"),
        default="depth",
        help="fill shallow depths first or select the highest-hit connected frontier",
    )
    parser.add_argument(
        "--sglang-python-root",
        type=Path,
        help="directory containing the sglang Python package (or set SGLANG_PYTHON_ROOT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="destination directory (default: artifacts)",
    )
    parser.add_argument(
        "--output-prefix",
        help="output filename prefix (default: trace filename without extension)",
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--tree-json-output", type=Path)
    parser.add_argument("--svg-output", type=Path)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--hit-distribution-csv", type=Path)
    parser.add_argument("--hit-distribution-svg", type=Path)
    parser.add_argument("--hit-distribution-png", type=Path)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip Graphviz and matplotlib images; still write JSON and CSV",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build, validate, export, and optionally render the SGLang radix tree."""
    args = parse_args(argv)
    sglang_python_root = discover_sglang_python_root(args.sglang_python_root)
    add_sglang_to_path(sglang_python_root)

    prefix = args.output_prefix or args.trace_file.stem
    output_dir = args.output_dir
    manifest_output = args.manifest_output or output_dir / f"{prefix}_manifest.json"
    tree_json_output = (
        args.tree_json_output or output_dir / f"{prefix}_radix_tree_full.json"
    )
    svg_output = args.svg_output or output_dir / f"{prefix}_radix_tree.svg"
    png_output = args.png_output or output_dir / f"{prefix}_radix_tree.png"
    hit_csv_output = (
        args.hit_distribution_csv
        or output_dir / f"{prefix}_hit_count_distribution.csv"
    )
    hit_svg_output = (
        args.hit_distribution_svg
        or output_dir / f"{prefix}_hit_count_distribution.svg"
    )
    hit_png_output = (
        args.hit_distribution_png
        or output_dir / f"{prefix}_hit_count_distribution.png"
    )
    (rows, block_sizes, total_hash_positions), selected_trace_format = (
        load_trace_with_format(args.trace_file, args.block_size, args.trace_format)
    )
    cache = build_sglang_radix_cache(rows)
    validation = validate_sglang_tree(cache, rows, block_sizes)
    depth_statistics = collect_depth_statistics(cache, len(rows))
    size_semantics = token_size_semantics(selected_trace_format)
    token_label = "est. tokens" if selected_trace_format == "lmcache_messages" else "tokens"
    nodes, displayed_by_depth = select_graph_nodes(
        cache,
        block_sizes,
        max_nodes=args.max_nodes,
        max_nodes_per_depth=args.max_nodes_per_depth,
        priority=args.priority,
    )

    dot = dot_text(nodes, len(rows), depth_statistics, token_label=token_label)
    write_json(
        full_tree_document(
            cache,
            block_sizes,
            len(rows),
            size_semantics=size_semantics,
        ),
        tree_json_output,
    )
    if not args.no_render:
        if shutil.which("dot") is None:
            raise RuntimeError("Graphviz 'dot' is required for rendering")
        render_graph(dot, svg_output, "svg")
        render_graph(dot, png_output, "png")

    hit_distribution = collect_hit_count_distribution(cache)
    write_hit_count_distribution_csv(hit_distribution, hit_csv_output)
    if not args.no_render:
        plot_hit_count_distribution(
            hit_distribution, [hit_svg_output, hit_png_output]
        )

    manifest = {
        "backend": "sglang.srt.mem_cache.radix_cache.RadixCache",
        "backend_mode": "create_simulated(page_size=1)",
        "backend_source_file": str(
            (
                sglang_python_root
                / "sglang/srt/mem_cache/radix_cache.py"
            ).resolve()
        ),
        "trace_file": str(args.trace_file.resolve()),
        "trace_format": selected_trace_format,
        "trace_sha256": hashlib.sha256(args.trace_file.read_bytes()).hexdigest(),
        "total_requests": len(rows),
        "total_hash_positions": total_hash_positions,
        "configured_block_size_tokens": args.block_size,
        "maximum_radix_unit_size_tokens": max(block_sizes.values()),
        "token_size_semantics": size_semantics,
        "unique_cache_blocks": len(block_sizes),
        "unique_cache_tokens": sum(block_sizes.values()),
        "node_usage_field": "TreeNode.hit_count",
        "graph_priority": args.priority,
        "selection": (
            "parent-preserving fair-share beam per radix depth: allocate slots "
            "round-robin across retained parents; rank siblings by hit_count"
        ),
        "max_graph_nodes": args.max_nodes,
        "max_nodes_per_depth": args.max_nodes_per_depth,
        "displayed_graph_nodes": len(nodes) + 1,
        "displayed_sglang_nodes_excluding_root": len(nodes),
        "displayed_nodes_by_depth": {
            str(depth): count for depth, count in displayed_by_depth.items()
        },
        "depth_statistics": {
            str(depth): asdict(stats)
            for depth, stats in depth_statistics.items()
        },
        "depth_percentile_method": "nearest-rank",
        "complete_tree_json_output": str(tree_json_output.resolve()),
        "svg_output": str(svg_output.resolve()) if not args.no_render else None,
        "png_output": str(png_output.resolve()) if not args.no_render else None,
        "hit_count_distribution": {
            "unique_hit_counts": len(hit_distribution),
            "minimum_hit_count": min(hit_distribution),
            "maximum_hit_count": max(hit_distribution),
            "nodes_with_hit_count_1": hit_distribution.get(1, 0),
            "nodes_with_hit_count_over_1": sum(
                count
                for hit_count, count in hit_distribution.items()
                if hit_count > 1
            ),
            "csv_output": (
                str(hit_csv_output.resolve())
            ),
            "svg_output": (
                str(hit_svg_output.resolve()) if not args.no_render else None
            ),
            "png_output": (
                str(hit_png_output.resolve()) if not args.no_render else None
            ),
        },
        **validation,
    }
    write_json(manifest, manifest_output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
