"""Separate c.11 hypothesis view, unchanged producer, observed trace and gate."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from . import dimension_association_v3 as v3
from .dimension_support_verified_v1 import propose
from .partial_ocr_consistency_v1 import verify,association_view
from .observation_completeness_v1 import gate
from .association_rejection_trace_v1 import RejectionTrace

ARMS=('unchanged','gate_only','consistency_and_gate')

def run(source_image,output,*,geometry,ocr,arrows,baseline,recognize):
    """No expected strings, annotations, manual view IDs or physical scales."""
    source_image=Path(source_image);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    digest,size,*_=v3._validate(source_image,geometry,ocr,arrows)
    if baseline['source_sha256']!=digest or baseline['processed_size_px']!=size:raise ValueError('Baseline frame mismatch')
    with Image.open(source_image) as image:gray=np.array(image.convert('L'))
    evidence=verify(gray,ocr,recognize);shadow=association_view(ocr,evidence)
    selected=shadow if evidence['groups'] and evidence['complete'] else ocr
    if selected is ocr:shadow_baseline=deepcopy(baseline)
    else:
        folder=output/'shadow-v3';folder.mkdir()
        shadow_baseline=v3.associate_dimensions_v3(source_image,folder,geometry=geometry,ocr=selected,arrows=arrows,record_timing=False)
    with RejectionTrace().observe() as before:
        raw=propose(source_image,geometry=geometry,ocr=ocr,arrows=arrows,baseline=baseline,arm='all')
    with RejectionTrace().observe() as after:
        grouped=propose(source_image,geometry=geometry,ocr=selected,arrows=arrows,baseline=shadow_baseline,arm='all')
    gated=gate(source_image,geometry=geometry,ocr=ocr,arrows=arrows,proposals=raw)
    grouped_gated=gate(source_image,geometry=geometry,ocr=selected,arrows=arrows,proposals=grouped)
    return {'algorithm_version':'dimension-chain-pipeline/1.0','source_sha256':digest,'processed_size_px':size,
            'arms':{'unchanged':raw,'gate_only':gated,'consistency_and_gate':grouped_gated},
            'consistency':evidence,'association_ocr_view':selected,'original_ocr_preserved':ocr,
            'traces':{'original':before.report(),'association_view':after.report()},
            'new_rec_calls':evidence['rec_calls'],'production_admitted':False,'physical_geometry_approved':False}
