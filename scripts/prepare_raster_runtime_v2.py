"""Bind the new input/service code after the source decoder qualification."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import read,pin,encoded
from cad_recovery.raster_stream_v2 import POLICY

FILES=('src/cad_recovery/raster_stream_v2.py','src/cad_recovery/raster_input_v2.py',
    'src/cad_recovery/profile_contract_v6.py','src/cad_recovery/profile_pipeline_v6.py',
    'src/cad_recovery/service_profiles_v11/__init__.py','src/cad_recovery/service_profiles_v11/api.py',
    'src/cad_recovery/service_profiles_v11/queue.py','src/cad_recovery/service_profiles_v11/worker.py',
    'scripts/service_profile_pipeline_v6.py','scripts/run_profiles_v13.py','scripts/convert_raster_v2.py',
    'scripts/prepare_raster_runtime_v2.py')


if __name__=='__main__':
    path=ROOT/'configs/raster-input-v2-runtime.json'
    if path.exists():raise ValueError('Runtime binding already exists')
    ref=pin(ROOT,'artifacts/raster-input-v2-qualification/summary.json')
    if not read(ROOT/ref['path'])['decoder_qualified']:raise ValueError('Decoder is unqualified')
    path.write_bytes(encoded(dict(policy=POLICY,qualification=ref,code=[pin(ROOT,p) for p in FILES],
        full_resolution_quality_accepted=False,old_profiles_unchanged=True)))
    print('Qualified bounded-input runtime frozen; old profile bindings preserved.')
