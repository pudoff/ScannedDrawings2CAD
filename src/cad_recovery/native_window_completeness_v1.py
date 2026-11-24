"""Authenticate every child before applying a gate to a merged native result."""
from pathlib import Path
from .dataset_layout_v1 import read
from .observation_completeness_v1 import assess,gate_native

def audit(folder):
    folder=Path(folder);report=read(folder/'observations.json');rows=[]
    for window in report['windows']:
        child=folder/window['id'];geometry=read(child/'page-001-geometry.json')
        ocr=read(child/'page-001-ocr.json');arrows=read(child/'page-001-arrows.json')
        baseline=read(child/'page-001-dimension-associations.json')
        if (window['proof']['source_sha256']!=report['source_sha256']
                or window['proof']['region_to_source']!=window['processed_to_native']
                or geometry['processed_size_px']!=[window['box'][2]-window['box'][0],window['box'][3]-window['box'][1]]):
            raise ValueError('Native window source or transform rebound')
        result=assess(child/geometry['source'],geometry=geometry,ocr=ocr,arrows=arrows,proposals=baseline)
        rows.append({'window_id':window['id'],'box':window['box'],'assessment':result})
    gated=gate_native(report,rows)
    return {'windows':rows,'merged_assessment':gated['completeness_gate'],'gated_observations':gated,
            'source_sha256':report['source_sha256'],'original_decisions_preserved_in_audit':True}
