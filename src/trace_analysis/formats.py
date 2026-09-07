"""Extensible input-format adapters for trace analysis."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One validated request path consumed by the radix-tree builder."""

    line_number: int
    hash_ids: tuple[int, ...]


TraceLoadResult = tuple[list[TraceRow], dict[int, int], int]
TraceLoader = Callable[[Path, int], TraceLoadResult]
TraceSniffer = Callable[[Path], bool]


@dataclass(frozen=True, slots=True)
class TraceAdapter:
    """A named parser that converts a source trace to normalized paths."""

    name: str
    suffixes: tuple[str, ...]
    loader: TraceLoader
    sniffer: TraceSniffer


_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(adapter: TraceAdapter) -> None:
    """Register an input adapter; intended as the extension point for formats."""
    if adapter.name in _ADAPTERS:
        raise ValueError(f"trace adapter {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter


def available_formats() -> tuple[str, ...]:
    """Return registered format names in deterministic order."""
    return tuple(sorted(_ADAPTERS))


def supported_suffixes() -> tuple[str, ...]:
    """Return all file suffixes advertised by registered adapters."""
    return tuple(sorted({suffix for adapter in _ADAPTERS.values() for suffix in adapter.suffixes}))


def token_size_semantics(trace_format: str) -> str:
    """Describe whether normalized node sizes are native or estimated."""
    if trace_format == "lmcache_messages":
        return "estimated_tokens_from_canonical_message_utf8_bytes_divided_by_4"
    return "native_trace_tokens"


def load_mooncake_trace(trace_file: Path, block_size: int) -> TraceLoadResult:
    """Parse Mooncake JSONL and retain exact final-block token sizes."""
    rows: list[TraceRow] = []
    block_sizes: dict[int, int] = {}
    total_hash_positions = 0

    with trace_file.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            hash_ids = tuple(record.get("hash_ids") or ())
            input_length = record.get("input_length")
            if not hash_ids or not isinstance(input_length, int):
                raise ValueError(
                    f"line {line_number}: hash_ids and integer input_length are required"
                )

            final_block_size = input_length - (len(hash_ids) - 1) * block_size
            if not 1 <= final_block_size <= block_size:
                raise ValueError(
                    f"line {line_number}: final block size {final_block_size} is outside "
                    f"[1, {block_size}]"
                )

            for index, hash_id in enumerate(hash_ids):
                if not isinstance(hash_id, int):
                    raise ValueError(f"line {line_number}: every hash_id must be an integer")
                size = final_block_size if index == len(hash_ids) - 1 else block_size
                previous = block_sizes.setdefault(hash_id, size)
                if previous != size:
                    raise ValueError(
                        f"line {line_number}: hash_id {hash_id} has conflicting token "
                        f"sizes {previous} and {size}"
                    )

            rows.append(TraceRow(line_number=line_number, hash_ids=hash_ids))
            total_hash_positions += len(hash_ids)

    if not rows:
        raise ValueError(f"trace is empty: {trace_file}")
    return rows, block_sizes, total_hash_positions


def detect_format(trace_file: Path) -> str:
    """Detect a registered trace format from its suffix and first record."""
    if trace_file.is_dir():
        shards = sorted(trace_file.rglob("*.parquet"))
        if shards and all(sniff_lmcache_messages(shard) for shard in shards):
            return "lmcache_messages"
        raise ValueError(f"directory does not contain supported LMCache shards: {trace_file}")
    suffix_matches = [
        adapter
        for adapter in _ADAPTERS.values()
        if trace_file.name.endswith(adapter.suffixes)
    ]
    matching = [adapter.name for adapter in suffix_matches if adapter.sniffer(trace_file)]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise ValueError(
            f"multiple trace formats match {trace_file}: " + ", ".join(matching)
        )
    raise ValueError(
        f"cannot auto-detect the format of {trace_file}; choose one of "
        + ", ".join(available_formats())
    )


def sniff_mooncake(trace_file: Path) -> bool:
    """Recognize Mooncake fields without consuming the full trace."""
    try:
        with trace_file.open("rb") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    return (
                        isinstance(record, dict)
                        and isinstance(record.get("input_length"), int)
                        and isinstance(record.get("hash_ids"), list)
                    )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def sniff_semianalysis_cc(trace_file: Path) -> bool:
    """Recognize the SemiAnalysis nested, local-hash trace format."""
    try:
        with trace_file.open("rb") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    return (
                        isinstance(record, dict)
                        and isinstance(record.get("block_size"), int)
                        and record.get("hash_id_scope") == "local"
                        and isinstance(record.get("requests"), list)
                    )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def load_semianalysis_cc_trace(
    trace_file: Path, _configured_block_size: int
) -> TraceLoadResult:
    """Flatten nested CC sessions and namespace their trace-local hash IDs."""
    rows: list[TraceRow] = []
    block_sizes: dict[int, int] = {}
    total_hash_positions = 0
    next_global_hash_id = 0
    request_number = 0

    with trace_file.open("rb") as stream:
        for source_line, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            block_size = record.get("block_size")
            requests = record.get("requests")
            if (
                not isinstance(block_size, int)
                or block_size < 1
                or not isinstance(requests, list)
            ):
                raise ValueError(
                    f"line {source_line}: block_size and requests are required"
                )

            # Hash IDs are meaningful only inside this top-level trace. Map
            # every local ID into one global integer namespace before combining
            # sessions in the SGLang tree.
            local_to_global: dict[tuple[str | None, int], int] = {}
            for request_index, request in enumerate(requests):
                request_number += 1
                hash_ids = request.get("hash_ids") if isinstance(request, dict) else None
                input_length = request.get("in") if isinstance(request, dict) else None
                model = request.get("model") if isinstance(request, dict) else None
                if not isinstance(hash_ids, list) or not hash_ids or not isinstance(input_length, int):
                    raise ValueError(
                        f"line {source_line}, request {request_index}: "
                        "hash_ids and integer in are required"
                    )
                final_block_size = input_length - (len(hash_ids) - 1) * block_size
                if not 1 <= final_block_size <= block_size:
                    raise ValueError(
                        f"line {source_line}, request {request_index}: final block "
                        f"size {final_block_size} is outside [1, {block_size}]"
                    )

                normalized_ids: list[int] = []
                for index, local_hash_id in enumerate(hash_ids):
                    if not isinstance(local_hash_id, int):
                        raise ValueError(
                            f"line {source_line}, request {request_index}: "
                            "every hash_id must be an integer"
                        )
                    scoped_hash_id = (model, local_hash_id)
                    if scoped_hash_id not in local_to_global:
                        local_to_global[scoped_hash_id] = next_global_hash_id
                        next_global_hash_id += 1
                    global_hash_id = local_to_global[scoped_hash_id]
                    size = (
                        final_block_size if index == len(hash_ids) - 1 else block_size
                    )
                    previous = block_sizes.setdefault(global_hash_id, size)
                    if previous != size:
                        raise ValueError(
                            f"line {source_line}, request {request_index}: local "
                            f"hash_id {local_hash_id} has conflicting token sizes "
                            f"{previous} and {size}"
                        )
                    normalized_ids.append(global_hash_id)

                rows.append(
                    TraceRow(
                        line_number=request_number,
                        hash_ids=tuple(normalized_ids),
                    )
                )
                total_hash_positions += len(normalized_ids)

    if not rows:
        raise ValueError(f"trace is empty: {trace_file}")
    return rows, block_sizes, total_hash_positions


def sniff_lmcache_messages(trace_file: Path) -> bool:
    """Recognize an LMCache message-history Parquet shard from its schema."""
    if trace_file.suffix != ".parquet":
        return False
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(trace_file).schema_arrow.names)
        return {"session_id", "model", "input", "output_length"} <= names
    except (ImportError, OSError):
        return False


def load_lmcache_messages(
    trace_file: Path, _configured_block_size: int
) -> TraceLoadResult:
    """Build deterministic message-prefix units from cumulative LMCache inputs.

    The source has no tokenizer output or KV hashes. Every canonical message is
    therefore one radix unit, namespaced by model and prefix. Its size is an
    explicit estimate of one token per four UTF-8 bytes.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for the lmcache_messages adapter"
        ) from error

    rows: list[TraceRow] = []
    block_sizes: dict[int, int] = {}
    total_hash_positions = 0
    edge_ids: dict[tuple[str, int, bytes], int] = {}
    next_edge_id = 0

    shards = sorted(trace_file.rglob("*.parquet")) if trace_file.is_dir() else [trace_file]

    def batches():
        for shard in shards:
            yield from pq.ParquetFile(shard).iter_batches(
                batch_size=64, columns=["model", "input"]
            )

    # Keep one prefix namespace across shards so shared messages accumulate hits.
    for batch in batches():
        for record in batch.to_pylist():
            model = record.get("model")
            messages = record.get("input")
            if not isinstance(model, str) or not isinstance(messages, list) or not messages:
                raise ValueError("LMCache rows require model and non-empty input fields")

            parent_edge_id = -1
            normalized_ids: list[int] = []
            for message in messages:
                canonical = json.dumps(
                    message,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                digest = hashlib.blake2b(canonical, digest_size=16).digest()
                edge_key = (model, parent_edge_id, digest)
                edge_id = edge_ids.get(edge_key)
                if edge_id is None:
                    edge_id = next_edge_id
                    next_edge_id += 1
                    edge_ids[edge_key] = edge_id
                    block_sizes[edge_id] = max(1, (len(canonical) + 3) // 4)
                normalized_ids.append(edge_id)
                parent_edge_id = edge_id

            rows.append(
                TraceRow(line_number=len(rows) + 1, hash_ids=tuple(normalized_ids))
            )
            total_hash_positions += len(normalized_ids)

    if not rows:
        raise ValueError(f"trace is empty: {trace_file}")
    return rows, block_sizes, total_hash_positions


def load_trace(
    trace_file: Path,
    block_size: int,
    trace_format: str = "auto",
) -> tuple[TraceLoadResult, str]:
    """Normalize a trace through the selected or auto-detected adapter."""
    selected_format = detect_format(trace_file) if trace_format == "auto" else trace_format
    try:
        adapter = _ADAPTERS[selected_format]
    except KeyError as error:
        raise ValueError(
            f"unknown trace format {selected_format!r}; available: "
            + ", ".join(available_formats())
        ) from error
    return adapter.loader(trace_file, block_size), selected_format


register_adapter(
    TraceAdapter(
        name="mooncake",
        suffixes=(".jsonl",),
        loader=load_mooncake_trace,
        sniffer=sniff_mooncake,
    )
)

register_adapter(
    TraceAdapter(
        name="lmcache_messages",
        suffixes=(".parquet",),
        loader=load_lmcache_messages,
        sniffer=sniff_lmcache_messages,
    )
)

register_adapter(
    TraceAdapter(
        name="semianalysis_cc",
        suffixes=(".jsonl",),
        loader=load_semianalysis_cc_trace,
        sniffer=sniff_semianalysis_cc,
    )
)
