"""Execute the frozen CV/OCR/native-CAD components on exact native pixels.

The adapter is a NEW candidate binding. It does not masquerade as profile v6:
there is no per-window deskew/resampling, and crop-local FRAME/GRID authority
is withdrawn when the crop is not the complete page.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from .dataset_layout_v1 import encoded, digest
from .grid_role_revision_v1 import sha


def frame_context(gray, context, box):
    x, y, _, _ = box
    ow, oh = context['size_px']
    w, h = context['source_size_px']
    sx, sy = w/ow, h/oh
    return dict(policy=context['policy'], context_sha256=context['pixel_sha256'],
                working_sha256=sha(gray), context_size_px=[ow, oh],
                working_size_px=list(gray.shape[::-1]),
                context_to_working=[[sx, 0, (sx-1)/2-x], [0, sy, (sy-1)/2-y], [0, 0, 1]],
                detector=context['frame_detector'], independent_semantic_accuracy_claimed=False,
                registration='authenticated_whole_page_overview_to_native_window')


def run_window(gray, folder, tile, context, engine):
    from .experimental_cv_v1 import analyze
    from .ocr import recognize_text
    from .native_annotation_inputs_v1 import load_inputs
    from . import native_cad_layers_candidate_v1 as candidate
    folder = Path(folder)
    folder.mkdir(parents=True)
    source = folder/'source.png'
    Image.fromarray(gray).save(source)
    # Observe OCR even if the CV path budget refuses the window.
    recognize_text(source, folder, engine=engine, prefix='native', time_budget_seconds=240.)
    observed = folder/'experimental'
    incomplete_reason = None
    try:
        result = analyze(source, observed)
    except ValueError as error:
        if str(error) != 'Incomplete source paths':
            raise
        incomplete_reason = str(error)
        report = dict(schema_version='native-cad-window/1.0', core_completed=False,
                      reason=incomplete_reason, source_sha256=digest(source),
                      source_pixels_exact=True, source_ink_fallback=True,
                      CAD=None, source_to_window=[[1, 0, -tile['box'][0]],
                                                 [0, 1, -tile['box'][1]], [0, 0, 1]])
        (folder/'window.json').write_bytes(encoded(report))
        return report
    if result['summary']['resized'] or not np.array_equal(gray, result['gray']):
        raise ValueError('Frozen core resized or altered native source pixels')
    root = Path(__file__).resolve().parents[2]
    packet, protection, annotations, _, _ = load_inputs(root, observed, folder/'native-ocr.json')
    whole = tile['box'] == [0, 0, *context['source_size_px']]
    withdrawn = int(np.isin(packet['roles'], [7, 8]).sum()) if not whole else 0
    if not whole:
        packet['roles'] = packet['roles'].copy()
        packet['roles'][np.isin(packet['roles'], [7, 8])] = 255
    Image.fromarray(packet['roles']).save(folder/'cad-authority-roles.png')
    destination = folder/'cad'
    destination.mkdir()
    evidence = frame_context(gray, context, tile['box'])
    # Dedicated single-job process, same isolation as the frozen CAD producer.
    with patch.object(candidate, 'working_context', lambda image: deepcopy(evidence)):
        exported = candidate.export_candidate(packet, annotations, destination/'annotated.dxf',
                                              protected=protection, render=False)
    report = dict(schema_version='native-cad-window/1.0',
        core_completed=bool(result['summary']['search_complete']),
        reason=None if result['summary']['search_complete'] else 'Frozen CV reported incomplete search',
        source_sha256=digest(source), source_pixels_exact=True, source_ink_fallback=False,
        withdrawn_crop_local_frame_grid_pixels=withdrawn, global_frame_context=evidence,
        source_to_window=[[1, 0, -tile['box'][0]], [0, 1, -tile['box'][1]], [0, 0, 1]],
        CAD=dict(path=str((destination/'annotated.dxf').resolve()), sha256=digest(destination/'annotated.dxf')),
        native_counts=exported['counts'], annotations_complete=exported['source_partition_exact'])
    (folder/'window.json').write_bytes(encoded(report))
    return report
