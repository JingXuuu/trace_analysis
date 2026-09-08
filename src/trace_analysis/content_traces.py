"""Adapters for content IDs, normalized into prefix-scoped radix edges."""
from __future__ import annotations

import json
from pathlib import Path

RAG_FILES = ('1_sys_prompt.jsonl', '2_passages.jsonl', '3_history.jsonl',
             '4_user_input.jsonl', '5_web_search.jsonl')
RAG_COMPONENTS = {'sys_prompt', 'passages_ids', 'history', 'user_input', 'web_search'}


def first_record(path):
    try:
        with path.open() as stream:
            record = next((json.loads(line) for line in stream if line.strip()), {})
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def sniff_bailian(path):
    record = first_record(path)
    return isinstance(record.get('hash_ids'), list) and 'chat_id' in record and 'input_length' in record


def sniff_ragpulse(path):
    record = first_record(path)
    hashes = record.get('hash_ids')
    return isinstance(hashes, dict) and set(hashes) == RAG_COMPONENTS


class PrefixUnits:
    """A repeated content ID under a different prefix is a different KV unit."""
    def __init__(self):
        self.edges = {}
        self.sizes = {}
        self.rows = []
        self.positions = 0

    def append(self, line, units):
        from .formats import TraceRow
        parent, path = -1, []
        for content, size in units:
            if not isinstance(size, int) or size < 0:
                raise ValueError(f'line {line}: invalid token length {size}')
            if size == 0:
                continue
            key = (parent, content, size)
            edge = self.edges.get(key)
            if edge is None:
                edge = len(self.edges)
                self.edges[key] = edge
                self.sizes[edge] = size
            path.append(edge)
            parent = edge
        if not path:
            raise ValueError(f'line {line}: empty request')
        self.rows.append(TraceRow(line, tuple(path)))
        self.positions += len(path)

    def result(self):
        if not self.rows:
            raise ValueError('trace contains no requests')
        return self.rows, self.sizes, self.positions


def load_bailian(path: Path, _block_size: int):
    tree = PrefixUnits()
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            record = json.loads(text)
            hashes, length = record['hash_ids'], record['input_length']
            if not isinstance(length, int) or length <= 0 or len(hashes) != (length + 15) // 16:
                raise ValueError(f'line {line}: Bailian requires ceil(input_length / 16) block IDs')
            if any(not isinstance(h, int) for h in hashes):
                raise ValueError(f'line {line}: block IDs must be integers')
            # A padded final block is not evidence that its partial prefix is
            # equivalent to another block. Include the used length in the key.
            tree.append(line, ((h, min(16, length - i * 16)) for i, h in enumerate(hashes)))
    return tree.result()


def load_ragpulse(path: Path, _block_size: int):
    lengths = {}
    for filename in RAG_FILES:
        with (path.parent / filename).open() as stream:
            for text in stream:
                if not text.strip():
                    continue
                record = json.loads(text)
                id_fields = [key for key in record if key != 'token_length']
                if len(id_fields) != 1 or not id_fields[0].endswith('_id'):
                    raise ValueError(f'{filename}: expected one content ID and token_length')
                content, length = record[id_fields[0]], record['token_length']
                if not isinstance(content, int) or not isinstance(length, int) or length < 0:
                    raise ValueError(f'{filename}: invalid content ID or token length')
                if content in lengths and lengths[content] != length:
                    raise ValueError(f'conflicting RAGPulse length for ID {content}')
                lengths[content] = length
    tree = PrefixUnits()
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            record = json.loads(text)
            if set(record['hash_ids']) != RAG_COMPONENTS:
                raise ValueError(f'line {line}: unsupported RAGPulse components')
            # Match the upstream replay: preserve component order in the JSON
            # object and the order of IDs in every component, never sort them.
            ids = [h for component in record['hash_ids'].values() for h in component]
            tree.append(line, ((h, lengths[h]) for h in ids))
    return tree.result()
