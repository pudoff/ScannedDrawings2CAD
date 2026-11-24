"""Versioned native-CAD import correction over the frozen window producer."""
from pathlib import Path
from unittest.mock import patch

from . import native_cad_pipeline_v1 as previous
from .native_cad_compose_v2 import Composer
from .dataset_layout_v1 import digest, encoded, read

CODE = ['src/cad_recovery/native_cad_pipeline_v1.py', 'src/cad_recovery/native_cad_pipeline_v2.py',
        'src/cad_recovery/native_cad_window_v1.py', 'src/cad_recovery/native_cad_compose_v1.py',
        'src/cad_recovery/native_cad_compose_v2.py', 'src/cad_recovery/native_cad_stitch_v1.py',
        'src/cad_recovery/native_cad_cache_v1.py', 'src/cad_recovery/native_sheet_source_v1.py',
        'src/cad_recovery/native_sheet_stitch_v1.py', 'src/cad_recovery/native_merge_v1.py']


def binding():
    root = Path(__file__).resolve().parents[2]
    from .profile_contract_v6 import resolve_profile
    return dict(schema_version='native-cad-adapter/2.0',
                frozen_component_binding=resolve_profile('native_cad_layers_draft'),
                code=[dict(path=f, sha256=digest(root/f)) for f in CODE],
                per_window_resampling=False, production_admitted=False)


def run(source, output, **kwargs):
    # Check before the finally block: a refused rerun must not overwrite the
    # binding sidecar of an existing job, including a failed one.
    if Path(output).exists():
        raise FileExistsError(output)
    contract = binding()
    try:
        with patch.object(previous, 'Composer', Composer):
            result = previous.run(source, output, **kwargs)
        result.update(schema_version='native-cad-pipeline/2.0', adapter_binding=contract)
        (Path(output)/'report.json').write_bytes(encoded(result))
        return result
    finally:
        if (Path(output)/'execution.json').exists():
            (Path(output)/'adapter-binding.json').write_bytes(encoded(contract))
