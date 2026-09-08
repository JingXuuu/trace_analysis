"""Pre-render default trees and package a standalone GitHub Pages website."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from importlib.resources import files
from pathlib import Path

from .reuse import reuse_path, source_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, default=Path('traces'))
    parser.add_argument('--charts-dir', type=Path, default=Path('artifacts/web'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs'))
    parser.add_argument('--force', action='store_true', help='Rebuild all default trees')
    args = parser.parse_args()
    from .web import AnalysisService
    from .cli import add_sglang_to_path, discover_sglang_python_root
    add_sglang_to_path(discover_sglang_python_root(None))
    output = args.output_dir.resolve()
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    service = AnalysisService([args.trace_dir], assets)
    catalog = []
    for item in sorted(service.traces(), key=lambda item: item['bytes']):
        path = service.resolve_trace(item['id'])
        chart = reuse_path(path, args.charts_dir)
        if not chart.is_file():
            raise RuntimeError(f'Prepare reuse charts first: ./scripts/prepare_reuse.sh ({path})')
        key = source_key(path)
        metadata = assets / f'tree-{key}.json'
        if metadata.exists() and not args.force:
            result = json.loads(metadata.read_text())
        else:
            print(f'Pre-rendering default tree: {item["name"]}', flush=True)
            result = service.plot({'trace': item['id'], 'max_nodes': 1000, 'max_nodes_per_depth': 16})
            for kind in ('svg', 'png'):
                rendered = assets / Path(result[f'{kind}_url']).name
                target = assets / f'tree-{key}.{kind}'
                rendered.replace(target)
                result[f'{kind}_url'] = f'./assets/{target.name}'
            result.pop('timings_seconds', None)
            metadata.write_text(json.dumps(result, indent=2) + '\n')
            service._cached_tree = None
            gc.collect()
        for kind in ('svg', 'png'):
            if not (output / result[f'{kind}_url']).is_file():
                raise RuntimeError(f'Missing pre-rendered asset for {metadata}')
        target_chart = assets / f'reuse-{key}.svg'
        shutil.copyfile(chart, target_chart)
        catalog.append({'id': str(len(catalog)), 'name': item['name'], 'bytes': item['bytes'],
                        'reuse_url': f'./assets/{target_chart.name}', 'plot': result})
        print(f'Ready: {item["name"]}', flush=True)
    html = files('trace_analysis').joinpath('static/index.html').read_text()
    assert 'const STATIC_MODE = false;' in html
    (output / 'index.html').write_text(html.replace('const STATIC_MODE = false;', 'const STATIC_MODE = true;'))
    (output / 'catalog.json').write_text(json.dumps({'traces': catalog}, indent=2) + '\n')
    (output / '.nojekyll').touch()
    print(f'Static site ready: {output}', flush=True)


if __name__ == '__main__':
    main()
