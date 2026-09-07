"""Dependency-free local web UI for interactive radix-tree plotting."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from .cli import (
    add_sglang_to_path,
    build_sglang_radix_cache,
    collect_depth_statistics,
    discover_sglang_python_root,
    dot_text,
    render_graph,
    select_graph_nodes,
    validate_sglang_tree,
)
from .formats import (
    available_formats,
    load_trace,
    supported_suffixes,
    token_size_semantics,
)
from .reuse import reuse_path


@dataclass(slots=True)
class CachedTree:
    """The last normalized trace and built SGLang tree."""

    key: tuple[Any, ...]
    cache: Any
    rows: list[Any]
    block_sizes: dict[int, int]
    validation: dict[str, int]
    depth_statistics: dict[int, Any]
    trace_format: str
    token_size_semantics: str


class AnalysisService:
    """Validate web settings, cache tree construction, and render graphs."""

    def __init__(self, trace_dirs: list[Path], output_dir: Path) -> None:
        self.trace_dirs = [directory.resolve() for directory in trace_dirs]
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cached_tree: CachedTree | None = None
        self._lock = threading.Lock()

    def traces(self) -> list[dict[str, Any]]:
        """List supported traces from every configured directory."""
        result: list[dict[str, Any]] = []
        seen: set[Path] = set()
        suffixes = supported_suffixes()
        for directory_index, directory in enumerate(self.trace_dirs):
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                dataset = next(
                    (parent for parent in path.parents
                     if parent.name == "lmcache-agentic-traces"
                     and parent.is_relative_to(directory)),
                    None,
                )
                if path.suffix == ".parquet" and dataset is not None:
                    path = dataset
                resolved = path.resolve()
                grouped = path.is_dir() and path.name == "lmcache-agentic-traces"
                if (
                    (not grouped and (not path.is_file() or not path.name.endswith(suffixes)))
                    or resolved in seen
                ):
                    continue
                members = sorted(path.rglob("*.parquet")) if grouped else [path]
                if not members:
                    continue
                seen.add(resolved)
                relative = path.relative_to(directory).as_posix()
                trace_id = f"{directory_index}:{relative}"
                reuse = reuse_path(path, self.output_dir)
                result.append(
                    {
                        "id": trace_id,
                        "name": relative,
                        "directory": str(directory),
                        "bytes": sum(member.stat().st_size for member in members),
                        "reuse_url": f"/generated/{reuse.name}" if reuse.is_file() else None,
                    }
                )
        return result

    def resolve_trace(self, trace_id: str) -> Path:
        """Resolve only an ID returned by the trace-list endpoint."""
        trace_map = {record["id"]: record for record in self.traces()}
        if trace_id not in trace_map:
            raise ValueError("select a trace from the current trace list")
        record = trace_map[trace_id]
        return (Path(record["directory"]) / record["name"]).resolve()

    @staticmethod
    def _integer_setting(
        payload: dict[str, Any], name: str, default: int, maximum: int
    ) -> int:
        raw_value = payload.get(name, default)
        if isinstance(raw_value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer") from error
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum:,}")
        return value

    def plot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build or reuse a full tree and render one selected view."""
        started = time.perf_counter()
        timings = dict.fromkeys(("load", "build", "validate", "statistics"), 0.0)

        @contextmanager
        def timed(stage):
            stage_start = time.perf_counter()
            try:
                yield
            finally:
                timings[stage] = time.perf_counter() - stage_start

        trace_file = self.resolve_trace(str(payload.get("trace", "")))
        max_nodes = self._integer_setting(payload, "max_nodes", 1000, 20_000)
        max_per_depth = self._integer_setting(
            payload, "max_nodes_per_depth", 16, 5_000
        )
        block_size = self._integer_setting(payload, "block_size", 512, 1_000_000)
        priority = "depth"
        trace_format = str(payload.get("trace_format", "auto"))
        if trace_format not in {"auto", *available_formats()}:
            raise ValueError("unknown trace format")

        members = sorted(trace_file.rglob("*.parquet")) if trace_file.is_dir() else [trace_file]
        cache_key = (
            trace_file,
            tuple((member, member.stat().st_mtime_ns, member.stat().st_size)
                  for member in members),
            block_size,
            trace_format,
        )
        timings["prepare"] = time.perf_counter() - started
        waiting = time.perf_counter()
        with self._lock:
            timings["queue"] = time.perf_counter() - waiting
            cache_hit = self._cached_tree is not None and self._cached_tree.key == cache_key
            if not cache_hit:
                with timed("load"):
                    (rows, block_sizes, _), selected_format = load_trace(
                        trace_file, block_size, trace_format
                    )
                with timed("build"):
                    cache = build_sglang_radix_cache(rows)
                with timed("validate"):
                    validation = validate_sglang_tree(cache, rows, block_sizes)
                with timed("statistics"):
                    depth_statistics = collect_depth_statistics(cache, len(rows))
                self._cached_tree = CachedTree(
                    key=cache_key,
                    cache=cache,
                    rows=rows,
                    block_sizes=block_sizes,
                    validation=validation,
                    depth_statistics=depth_statistics,
                    trace_format=selected_format,
                    token_size_semantics=token_size_semantics(selected_format),
                )

            tree = self._cached_tree
            selection_start = time.perf_counter()
            nodes, displayed_by_depth = select_graph_nodes(
                tree.cache,
                tree.block_sizes,
                max_nodes=max_nodes,
                max_nodes_per_depth=max_per_depth,
                priority=priority,
            )
            timings["selection"] = time.perf_counter() - selection_start
            layout_start = time.perf_counter()
            token_label = (
                "est. tokens" if tree.trace_format == "lmcache_messages" else "tokens"
            )
            dot = dot_text(
                nodes,
                len(tree.rows),
                tree.depth_statistics,
                token_label=token_label,
            )
            timings["graph_description"] = time.perf_counter() - layout_start
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_file.stem)
            output_stem = (
                f"{safe_stem}_{priority}_nodes-{max_nodes}_depth-{max_per_depth}"
            )
            svg_file = self.output_dir / f"{output_stem}.svg"
            png_file = self.output_dir / f"{output_stem}.png"
            with timed("svg"):
                render_graph(dot, svg_file, "svg")
            with timed("png"):
                render_graph(dot, png_file, "png")

        result = {
            "trace": trace_file.name,
            "trace_format": tree.trace_format,
            "priority": priority,
            "maximum_radix_unit_size_tokens": max(tree.block_sizes.values()),
            "token_size_semantics": tree.token_size_semantics,
            "total_requests": len(tree.rows),
            "full_tree_nodes": tree.validation[
                "validated_sglang_nodes_excluding_root"
            ]
            + 1,
            "displayed_nodes": len(nodes) + 1,
            "displayed_nodes_by_depth": displayed_by_depth,
            "svg_url": f"/generated/{svg_file.name}",
            "png_url": f"/generated/{png_file.name}",
        }
        timings["total"] = time.perf_counter() - started
        result["timings_seconds"] = timings
        result["tree_cache_hit"] = cache_hit
        return result


class RequestHandler(BaseHTTPRequestHandler):
    """Serve the single-page UI and its small JSON API."""

    service: AnalysisService
    index_html: bytes

    def _send_bytes(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, document: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(document).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(self.index_html, "text/html; charset=utf-8")
            return
        if path == "/api/traces":
            self._send_json(
                {
                    "traces": self.service.traces(),
                    "formats": ["auto", *available_formats()],
                }
            )
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if path.startswith("/generated/"):
            filename = unquote(path.removeprefix("/generated/"))
            candidate = (self.service.output_dir / filename).resolve()
            if (
                "/" in filename
                or not candidate.is_relative_to(self.service.output_dir)
                or not candidate.is_file()
                or candidate.suffix not in {".svg", ".png"}
            ):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = "image/svg+xml" if candidate.suffix == ".svg" else "image/png"
            self._send_bytes(candidate.read_bytes(), content_type)
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/plot":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 64 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            self._send_json(self.service.plot(payload))
        except (ValueError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # keep the browser useful without leaking a traceback
            self.log_error("plot failed: %s", error)
            self._send_json(
                {"error": f"analysis failed: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the trace radix-tree web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--trace-dir",
        action="append",
        type=Path,
        dest="trace_dirs",
        help="trace directory; repeat to add sources (default: ./traces)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/web"))
    parser.add_argument("--sglang-python-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' is required by the web plotter")
    sglang_root = discover_sglang_python_root(args.sglang_python_root)
    add_sglang_to_path(sglang_root)
    trace_dirs = args.trace_dirs or [Path("traces")]
    RequestHandler.service = AnalysisService(trace_dirs, args.output_dir)
    RequestHandler.index_html = (
        files("trace_analysis").joinpath("static/index.html").read_bytes()
    )
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Trace Radix Analysis: http://{args.host}:{args.port}", flush=True)
    print(
        "Trace directories: "
        + ", ".join(str(directory.resolve()) for directory in trace_dirs),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
