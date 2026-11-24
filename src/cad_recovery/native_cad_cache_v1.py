"""Explicit cleanup of owned failed native-CAD caches; original input is retained."""
from pathlib import Path
import shutil

from .dataset_layout_v1 import read, encoded


def cleanup_failed(output):
    output = Path(output).resolve()
    state_file = output/'execution.json'
    state = read(state_file)
    if state.get('schema_version') != 'native-cad-pipeline/1.0' or state.get('state') not in ('failed', 'cancelled'):
        raise ValueError('Only an explicitly named failed/cancelled native CAD run can be cleaned')
    if list(output.glob('page-*.dxf')) or (output/'report.json').exists():
        raise ValueError('Published results are retained; failed-cache cleanup refused')
    source = None
    if (output/'source/manifest.json').exists():
        source = Path(read(output/'source/manifest.json')['source']['path']).resolve()
    targets = [p for p in [output/'source', *output.glob('page-[0-9][0-9][0-9]')] if p.exists()]
    # Validate ALL absolute targets before the first recursive operation.
    for path in targets:
        resolved = path.resolve()
        if resolved == output or not resolved.is_relative_to(output) or path.is_symlink():
            raise ValueError('Cache target escapes the explicitly named run directory')
        if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()) for p in [path, *path.rglob('*')]):
            raise ValueError('Cache contains symbolic links/junctions; cleanup refused')
        if source and source.is_relative_to(resolved):
            raise ValueError('Original input resides inside cache; cleanup refused')
    removed = []
    for path in targets:
        removed.append(path.name)
        shutil.rmtree(path)
    state.update(cache_cleaned=True, removed_private_paths=removed,
                 original_source_retained=source is None or source.exists())
    state_file.write_bytes(encoded(state))
    return state
