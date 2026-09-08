from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace_analysis.cli import (
    collect_hit_count_distribution,
    collect_depth_statistics,
    dot_text,
    full_tree_document,
    load_trace,
    select_graph_nodes,
)
from trace_analysis.formats import detect_format, token_size_semantics
from trace_analysis.web import AnalysisService
from trace_analysis.reuse import reuse_path, group_reuse

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


class FakeNode:
    _next_id = 0

    def __init__(self, key: list[int], hit_count: int) -> None:
        self.id = FakeNode._next_id
        FakeNode._next_id += 1
        self.key = key
        self.hit_count = hit_count
        self.children: dict[int, FakeNode] = {}

    def add(self, node: "FakeNode") -> "FakeNode":
        self.children[node.key[0]] = node
        return node


class FakeCache:
    def __init__(self, root: FakeNode) -> None:
        self.root_node = root


def sample_cache() -> FakeCache:
    FakeNode._next_id = 0
    root = FakeNode([], 0)
    left = root.add(FakeNode([1], 10))
    right = root.add(FakeNode([2], 8))
    left.add(FakeNode([3], 7))
    left.add(FakeNode([4], 2))
    right.add(FakeNode([5], 6))
    right.add(FakeNode([6], 1))
    return FakeCache(root)


class AnalysisTests(unittest.TestCase):
    def test_node_reuse_counts_and_preserves_empty_bins(self) -> None:
        nodes = collect_hit_count_distribution(sample_cache())
        self.assertEqual(sum(nodes.values()), 6)
        self.assertEqual(group_reuse({1: 2, 8: 1}),
                         (['1', '2–3', '4–7', '8–15'], [2, 0, 0, 1]))

    def test_offline_reuse_discovery_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / 'trace.jsonl'
            trace.write_text('{"input_length":512,"hash_ids":[1]}\n')
            service = AnalysisService([root], root / 'plots')
            self.assertIsNone(service.traces()[0]['reuse_url'])
            reuse = reuse_path(trace, service.output_dir)
            reuse.write_text('<svg/>')
            self.assertEqual(service.traces()[0]['reuse_url'], f'/generated/{reuse.name}')
            trace.write_text('{"input_length":512,"hash_ids":[1]}\n' * 2)
            self.assertIsNone(service.traces()[0]['reuse_url'])

    @unittest.skipUnless(pa is not None and pq is not None, "pyarrow is optional")
    def test_lmcache_shards_form_one_trace_with_shared_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "lmcache-agentic-traces"
            data = dataset / "data"
            data.mkdir(parents=True)
            system = {"role": "system", "content": "Shared system prompt"}
            for index in range(2):
                record = {
                    "session_id": str(index), "model": "model-a",
                    "input": [system, {"role": "user", "content": str(index)}],
                    "output_length": 10,
                }
                pq.write_table(pa.Table.from_pylist([record]), data / f"train-{index}.parquet")
            service = AnalysisService([root, dataset], root / "outputs")
            catalog = service.traces()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(service.resolve_trace(catalog[0]["id"]), dataset)
            self.assertEqual(catalog[0]["bytes"], sum(p.stat().st_size for p in data.iterdir()))
            self.assertEqual(detect_format(dataset), "lmcache_messages")
            rows, sizes, positions = load_trace(dataset, 512)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].hash_ids[0], rows[1].hash_ids[0])
            self.assertNotEqual(rows[0].hash_ids[1], rows[1].hash_ids[1])
            self.assertEqual(len(sizes), 3)
            self.assertEqual(positions, 4)

    def test_trace_validation_retains_partial_final_block_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text(
                json.dumps({"input_length": 700, "hash_ids": [1, 2]}) + "\n",
                encoding="utf-8",
            )
            rows, block_sizes, positions = load_trace(trace, block_size=512)

        self.assertEqual(rows[0].hash_ids, (1, 2))
        self.assertEqual(block_sizes, {1: 512, 2: 188})
        self.assertEqual(positions, 2)

    def test_semianalysis_local_hashes_are_namespaced(self) -> None:
        records = [
            {
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [
                    {"in": 100, "hash_ids": [0, 1]},
                    {"in": 128, "hash_ids": [0, 2]},
                ],
            },
            {
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [{"in": 100, "hash_ids": [0, 1]}],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "cc.jsonl"
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(detect_format(trace), "semianalysis_cc")
            rows, block_sizes, positions = load_trace(trace, block_size=512)

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(block_sizes), 5)
        self.assertNotEqual(rows[0].hash_ids[0], rows[2].hash_ids[0])
        self.assertEqual(max(block_sizes.values()), 64)
        self.assertEqual(positions, 6)

    @unittest.skipUnless(pa is not None and pq is not None, "pyarrow is optional")
    def test_lmcache_parquet_builds_model_scoped_message_prefixes(self) -> None:
        system = {"role": "system", "content": "Be helpful."}
        user = {"role": "user", "content": "Inspect the trace."}
        records = [
            {
                "session_id": "a",
                "model": "model-a",
                "input": [system],
                "output_length": 10,
                "pre_gap": 0.0,
            },
            {
                "session_id": "a",
                "model": "model-a",
                "input": [system, user],
                "output_length": 10,
                "pre_gap": 0.1,
            },
            {
                "session_id": "b",
                "model": "model-b",
                "input": [system],
                "output_length": 10,
                "pre_gap": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "lmcache.parquet"
            pq.write_table(pa.Table.from_pylist(records), trace)
            self.assertEqual(detect_format(trace), "lmcache_messages")
            rows, block_sizes, positions = load_trace(trace, block_size=512)

        self.assertEqual(rows[0].hash_ids, rows[1].hash_ids[:1])
        self.assertNotEqual(rows[0].hash_ids, rows[2].hash_ids)
        self.assertEqual(len(block_sizes), 3)
        self.assertEqual(positions, 4)
        self.assertTrue(token_size_semantics("lmcache_messages").startswith("estimated"))

    def test_selection_keeps_each_selected_parent_connected(self) -> None:
        cache = sample_cache()
        nodes, by_depth = select_graph_nodes(
            cache,
            {key: 512 for key in range(1, 7)},
            max_nodes=1000,
            max_nodes_per_depth=2,
        )

        depth_one_ids = {node.graph_id for node in nodes if node.depth == 1}
        depth_two_parents = {
            node.parent_graph_id for node in nodes if node.depth == 2
        }
        self.assertEqual(depth_two_parents, depth_one_ids)
        self.assertEqual(by_depth, {1: 2, 2: 2})

    def test_graph_limits_include_root_and_cap_each_depth(self) -> None:
        nodes, by_depth = select_graph_nodes(
            sample_cache(),
            {key: 512 for key in range(1, 7)},
            max_nodes=4,
            max_nodes_per_depth=2,
        )
        self.assertLessEqual(len(nodes) + 1, 4)
        self.assertTrue(all(count <= 2 for count in by_depth.values()))

    def test_hit_count_priority_stays_connected(self) -> None:
        nodes, by_depth = select_graph_nodes(
            sample_cache(),
            {key: 512 for key in range(1, 7)},
            max_nodes=4,
            max_nodes_per_depth=4,
            priority="hit_count",
        )
        selected_ids = {"ROOT", *(node.graph_id for node in nodes)}
        self.assertEqual(len(nodes) + 1, 4)
        self.assertTrue(
            all(node.parent_graph_id in selected_ids for node in nodes)
        )
        self.assertTrue(all(count <= 4 for count in by_depth.values()))

    def test_depth_budget_is_shared_evenly_across_parents(self) -> None:
        FakeNode._next_id = 0
        root = FakeNode([], 0)
        parents = [root.add(FakeNode([index + 1], 100 - index)) for index in range(4)]
        block_sizes = {index + 1: 512 for index in range(4)}
        next_key = 10
        for parent_index, parent in enumerate(parents):
            for child_index in range(8):
                key = next_key
                next_key += 1
                parent.add(FakeNode([key], 1000 - parent_index * 100 - child_index))
                block_sizes[key] = 512

        nodes, _ = select_graph_nodes(
            FakeCache(root),
            block_sizes,
            max_nodes=1000,
            max_nodes_per_depth=16,
        )
        depth_one_ids = [node.graph_id for node in nodes if node.depth == 1]
        children_per_parent = {
            parent_id: sum(
                node.depth == 2 and node.parent_graph_id == parent_id for node in nodes
            )
            for parent_id in depth_one_ids
        }
        self.assertEqual(set(children_per_parent.values()), {4})

    def test_full_tree_and_distribution_cover_every_node(self) -> None:
        cache = sample_cache()
        document = full_tree_document(
            cache, {key: 512 for key in range(1, 7)}, total_requests=18
        )
        distribution = collect_hit_count_distribution(cache)

        self.assertEqual(len(document["nodes"]), 7)
        self.assertEqual(sum(distribution.values()), 6)
        ids = {record["id"] for record in document["nodes"]}
        self.assertTrue(
            all(
                record["parent_id"] in ids
                for record in document["nodes"]
                if record["parent_id"] is not None
            )
        )

    def test_shared_leaf_is_blue(self) -> None:
        cache = sample_cache()
        nodes, _ = select_graph_nodes(
            cache,
            {key: 512 for key in range(1, 7)},
            max_nodes=1000,
            max_nodes_per_depth=16,
        )
        dot = dot_text(nodes, total_requests=18)
        shared_leaf = next(node for node in nodes if node.is_leaf and node.hit_count > 1)
        line = next(line for line in dot.splitlines() if line.lstrip().startswith(f"{shared_leaf.graph_id} "))
        self.assertIn('fillcolor="#2b6cb0"', line)

    def test_depth_statistics_and_dashed_columns(self) -> None:
        cache = sample_cache()
        stats = collect_depth_statistics(cache, total_requests=18)
        self.assertEqual(stats[1].total_nodes, 2)
        self.assertEqual(stats[1].parent_nodes, 2)
        self.assertEqual(stats[1].sink_nodes, 0)
        self.assertEqual(stats[2].p50_hit_count, 2)
        self.assertEqual(stats[2].p99_hit_count, 7)

        nodes, _ = select_graph_nodes(
            cache,
            {key: 512 for key in range(1, 7)},
            max_nodes=1000,
            max_nodes_per_depth=16,
        )
        dot = dot_text(nodes, total_requests=18, depth_statistics=stats)
        self.assertIn("subgraph cluster_depth_2", dot)
        self.assertIn("style=dashed", dot)
        self.assertIn("parent nodes=0", dot)
        self.assertIn("sink nodes=4", dot)


if __name__ == "__main__":
    unittest.main()
