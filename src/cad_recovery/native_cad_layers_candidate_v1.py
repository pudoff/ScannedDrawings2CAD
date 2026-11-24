"""Source-supported FRAME assignments over the qualified CAD/OCR candidate.

Use in a dedicated single-job child, like native_cad_candidate_v1.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import cad_axis_candidate_export_v1 as geometry_export
from .grid_role_revision_v1 import sha
from .geometry_preserving_role_export_v1 import verify_partition
from .native_annotation_export_v2 import export_annotations
from .page_frame_semantics_v1 import apply, POLICY
from .page_frame_context_v3 import detect
from .stroke_roles_v1 import checked_gray


def working_context(gray):
    """The existing loader already binds this whole working page to the source.

    No second resize, even-padding or AREA/LANCZOS interchange is introduced.
    Odd dimensions are legitimate in the Web observation contract.
    """
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels']:
        raise ValueError('Frame context pixel budget exceeded')
    return dict(policy=deepcopy(POLICY), context_sha256=sha(gray), working_sha256=sha(gray),
        context_size_px=list(gray.shape[::-1]), working_size_px=list(gray.shape[::-1]),
        context_to_working=np.eye(3).tolist(), detector=detect(gray),
        independent_semantic_accuracy_claimed=False, registration='identity_whole_working_source')


def export_candidate(packet, annotations, target, *, protected=None, render=True):
    evidence = working_context(packet['gray'])
    captured = []
    original = geometry_export.role_guard

    def export_geometry(data, destination, *, protected=None, render=False):
        def guarded(assembly, gray, geometry, roles):
            baseline = original(assembly, gray, geometry, roles)
            result = apply(baseline, gray, geometry, roles, evidence, protected=protected)
            captured.append(dict(context=evidence, assignments=result['whole_page_frame_semantics'],
                                 partition=verify_partition(baseline, result)))
            return result
        with patch.object(geometry_export, 'role_guard', guarded):
            return geometry_export.export_objects(data, destination, protected=protected, render=render)

    report = export_annotations(packet, annotations, target, protected=protected, render=render,
                                geometry_export=export_geometry)
    if len(captured) != 1:
        raise ValueError('Exactly one frame assignment pass required')
    proof = captured[0]
    report['frame_semantics'] = proof
    Path(target).with_name('frame-semantics.json').write_text(
        json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report
