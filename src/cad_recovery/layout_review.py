"""Bounded diagnostic review of native paperspace layouts and viewports.

This does not replace a modelspace pair verdict or establish one global CAD
mapping: each viewport has its own model-to-paper mapping and clipping region.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random

import ezdxf
from ezdxf.addons.drawing import RenderContext, config, layout
from ezdxf.addons.drawing.pymupdf import PyMuPdfRenderBackend, get_coordinate_output_space
import numpy as np
import pymupdf

from .clear_pairs import digest
from .dxf_reference import (MAX_ENTITIES, MAX_FILE_BYTES, HATCH_JITTER_SEED,
    _HATCH_RANDOM_LOCK, _Recorder, _matrix_xy, normalize_reference)
from .dxf_reference_v5 import _Frontend as ReferenceFrontend
from .mtext_zero_height import install_zero_height_adapter
from .reference_completeness import apply_resource_completeness

METHOD = 'native-paperspace-viewports-diagnostic-v1'
MAX_LAYOUTS = 64
MAX_VIEWPORTS = 128
MAX_TEXT_RECORDS = 10000


def _read(path):
    path = Path(path).resolve()
    if path.suffix.lower() != '.dxf' or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('Layout review requires a DXF no larger than 50 MiB')
    before = digest(path)
    document = ezdxf.readfile(path)
    if len(document.entitydb) > MAX_ENTITIES or len(document.layouts) > MAX_LAYOUTS:
        raise ValueError('Layout review exceeds document budget')
    return path, before, document


def _viewport(vp, role):
    dxf = vp.dxf
    values = {name: list(dxf.get(name, default)) for name, default in (
        ('center', (0, 0, 0)), ('view_center_point', (0, 0)),
        ('view_target_point', (0, 0, 0)), ('view_direction_vector', (0, 0, 1)))}
    width, height, view_height = float(dxf.width), float(dxf.height), float(dxf.view_height)
    twist, flags = float(dxf.view_twist_angle), int(dxf.flags)
    unsupported = []
    if not all(math.isfinite(v) for seq in values.values() for v in seq) or not all(
            math.isfinite(v) for v in (width, height, view_height, twist)):
        unsupported.append('nonfinite_viewport')
    if min(width, height, view_height) <= 0:
        unsupported.append('nonpositive_viewport_extent')
    if not vp.is_top_view:
        unsupported.append('non_top_view')
    if flags & 1:
        unsupported.append('perspective_view')
    if twist != 0:
        unsupported.append('twisted_view_requires_separate_proof')
    if vp.has_extended_clipping_path:
        unsupported.append('nonrectangular_clip_requires_separate_proof')
    row = {'handle': dxf.handle, 'owner_handle': dxf.owner, 'layer': dxf.layer,
        'id': int(dxf.id), 'status': int(dxf.status), 'role': role,
        **values, 'width': width, 'height': height, 'view_height': view_height,
        'view_twist_angle': twist, 'flags': flags,
        'frozen_layers': list(vp.frozen_layers),
        'extended_clip': bool(vp.has_extended_clipping_path),
        'clip_boundary_handle': dxf.get('clipping_boundary_handle', '0'),
        'unsupported_reasons': unsupported,
        'physical_scale_confirmed': False}
    if not unsupported:
        scale = height/view_height
        center, view, base = values['center'], values['view_center_point'], values['view_target_point']
        # Independent top-view, zero-twist XY derivation; the library matrix
        # is checked, rather than treated as a geometric oracle.
        matrix = np.array([[scale, 0, center[0]-scale*(view[0]+base[0])],
                           [0, scale, center[1]-scale*(view[1]+base[1])], [0, 0, 1.]])
        if not np.allclose(matrix, _matrix_xy(vp.get_transformation_matrix()), rtol=1e-12, atol=1e-9):
            raise ValueError('Native viewport transform differs from independent planar derivation')
        cx, cy = center[:2]
        paper_clip = [cx-width/2, cy-height/2, cx+width/2, cy+height/2]
        inverse = np.linalg.inv(matrix)
        corners = np.array([[paper_clip[0], paper_clip[1], 1.], [paper_clip[2], paper_clip[3], 1.]])
        model_corners = (inverse @ corners.T).T
        row.update(model_to_paper=matrix.tolist(), paper_to_model=inverse.tolist(),
            declared_model_to_paper_scale=scale, paper_clip_xy=paper_clip,
            model_window_xy=[*model_corners[0, :2], *model_corners[1, :2]])
    return row


def _layout_inventory(document):
    rows = []
    for space in sorted(document.layouts, key=lambda item: (item.name != 'Model', item.name)):
        entities = list(space)
        viewports = [entity for entity in entities if entity.dxftype() == 'VIEWPORT']
        if len(viewports) > MAX_VIEWPORTS:
            raise ValueError('Layout viewport count exceeds budget')
        visible = sorted((vp for vp in viewports if vp.dxf.status > 0), key=lambda vp: vp.dxf.status)
        # Match the pinned frontend's documented active-view removal exactly.
        active = visible.pop(0) if visible and visible[0].dxf.status == 1 else None
        content_handles = {vp.dxf.handle for vp in visible}
        text = []
        for entity in entities:
            candidates = [entity, *entity.attribs] if entity.dxftype() == 'INSERT' else [entity]
            for candidate in candidates:
                if candidate.dxftype() not in ('TEXT', 'MTEXT', 'ATTRIB', 'ATTDEF'):
                    continue
                if len(text) >= MAX_TEXT_RECORDS:
                    raise ValueError('Layout text inventory exceeds budget')
                text.append({'handle': candidate.dxf.handle, 'type': candidate.dxftype(),
                    'owner_handle': candidate.dxf.owner, 'layer': candidate.dxf.layer,
                    'insert': list(candidate.dxf.insert),
                    'native_text': candidate.text if candidate.dxftype() == 'MTEXT' else candidate.dxf.text,
                    'plain_text': candidate.plain_text(),
                    'tag': candidate.dxf.get('tag') if candidate.dxftype() in ('ATTRIB', 'ATTDEF') else None})
        vp_rows = [_viewport(vp, 'layout_camera' if vp is active else
                    'content_viewport' if vp.dxf.handle in content_handles else 'inactive_viewport') for vp in viewports]
        dxf = space.dxf_layout.dxf
        paper_keys = ('paper_width', 'paper_height', 'plot_paper_units', 'plot_rotation', 'plot_type',
                      'scale_numerator', 'scale_denominator', 'standard_scale_type')
        rows.append({'name': space.name, 'is_modelspace': space.name == 'Model',
            'block_record_handle': space.block_record_handle, 'entity_count': len(entities),
            'entity_types': dict(Counter(e.dxftype() for e in entities)), 'viewports': vp_rows,
            'content_viewport_count': len(content_handles),
            'plot_settings': {key: dxf.get(key) for key in paper_keys},
            'native_text_records': text,
            'text_scope': 'Direct layout TEXT/MTEXT/ATTDEF and attached INSERT ATTRIB; not flattened block text or approved OCR.'})
    return rows


def audit_native_layouts(path):
    path, source_sha, document = _read(path)
    result = {'schema_version': '1.0', 'method': METHOD, 'source_sha256': source_sha,
        'layouts': _layout_inventory(document), 'source_modified': False,
        'training_eligible': False, 'physical_scale_confirmed': False,
        'limitations': ['Native viewport declarations do not prove PNG correspondence.',
            'Different viewports require separate model-to-paper transforms; a sheet has no single global modelspace transform.',
            'Viewport statuses follow the pinned frontend convention; authoring-CAD plotting behavior is not asserted.']}
    if digest(path) != source_sha:
        raise ValueError('DXF changed during native layout audit')
    return result


class _PaperFrontend(ReferenceFrontend):
    def __init__(self, *args, **kwargs):
        self.current_review_viewport = None
        self.drawn_viewports = []
        super().__init__(*args, **kwargs)

    def draw_entities_callback(self, ctx, entities):
        # Resource skip visibility must use the viewport's frozen-layer context.
        previous = self.ctx
        self.ctx = ctx
        try:
            return super().draw_entities_callback(ctx, entities)
        finally:
            self.ctx = previous

    def draw_viewport(self, vp):
        previous = self.current_review_viewport
        self.current_review_viewport = vp.dxf.handle
        self.drawn_viewports.append(vp.dxf.handle)
        try:
            return super().draw_viewport(vp)
        finally:
            self.current_review_viewport = previous

    def _resource_identity(self, entity):
        return {**super()._resource_identity(entity), 'paperspace_viewport_handle': self.current_review_viewport}

    def skip_entity(self, entity, msg):
        before = len(self.resource_skips)
        result = super().skip_entity(entity, msg)
        for row in self.resource_skips[before:]:
            row['paperspace_viewport_handle'] = self.current_review_viewport
        return result


def render_paperspace_diagnostic(path, layout_name, *, max_side=1600, resource_root=None):
    """Render one explicitly selected supported native sheet; never approve a pair."""
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 320 <= max_side <= 2400:
        raise ValueError('max_side must be an integer in 320..2400')
    path, source_sha, document = _read(path)
    inventory = _layout_inventory(document)
    selected = next((row for row in inventory if row['name'] == layout_name), None)
    if selected is None or selected['is_modelspace']:
        raise ValueError('Choose an existing named paperspace layout')
    content = [vp for vp in selected['viewports'] if vp['role'] == 'content_viewport']
    if not content or any(vp['unsupported_reasons'] for vp in content):
        raise ValueError('Paperspace diagnostic requires supported planar rectangular content viewports')
    audit = document.audit()
    if audit.errors:
        raise ValueError('DXF contains unresolved audit errors')
    zero_policy = install_zero_height_adapter(document)
    context, recorder = RenderContext(document), _Recorder()
    options = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
        color_policy=config.ColorPolicy.COLOR, hatching_timeout=5.0)
    frontend = _PaperFrontend(context, recorder, config=options)
    frontend.configure_resources(path, Path(resource_root).resolve() if resource_root else path.parent,
                                 allow_incomplete_reference=True)
    with _HATCH_RANDOM_LOCK:
        state = random.getstate()
        try:
            random.seed(HATCH_JITTER_SEED)
            frontend.draw_layout(document.layouts.get(layout_name), finalize=True)
        finally:
            random.setstate(state)
    if set(frontend.drawn_viewports) != {vp['handle'] for vp in content}:
        raise ValueError('Rendered viewport inventory differs from native selection')
    player = recorder.player()
    bounds = player.bbox()
    if not bounds.has_data or min(bounds.size) <= 0 or not np.isfinite(tuple(bounds.extmin)+tuple(bounds.extmax)).all():
        raise ValueError('Paper drawing bounds are empty, degenerate or nonfinite')
    paper_bounds = [*bounds.extmin, *bounds.extmax]
    page = layout.Page(297, 297, margins=layout.Margins.all(3))
    settings = layout.Settings(output_coordinate_space=get_coordinate_output_space(page))
    placement = layout.Layout(bounds, flip_y=True).get_placement_matrix(page, settings=settings, top_origin=True)
    paper_to_page = _matrix_xy(placement)
    player.transform(placement)
    backend = PyMuPdfRenderBackend(page, settings)
    try:
        player.replay(backend)
        page_size = [backend.page_width_in_pt, backend.page_height_in_pt]
        scale = max_side*1.5/max(page_size)
        pix = backend.page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
        canvas = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        page_to_canvas = np.array([[scale, 0., -pix.x-.5], [0., scale, -pix.y-.5], [0., 0., 1.]])
    finally:
        backend.doc.close()
    pixels, normalization = normalize_reference(canvas, max_side)
    paper_to_pixel = np.array(normalization['input_to_normalized']) @ page_to_canvas @ paper_to_page
    viewports = [{**vp, 'model_to_diagnostic_pixel': (paper_to_pixel @ np.array(vp['model_to_paper'])).tolist()}
                 for vp in content]
    resource_meta = apply_resource_completeness({'source_sha256': source_sha,
        'reference_complete': frontend.reference_complete, 'resource_observations': frontend.resource_observations,
        'resource_skip_audit_complete': True, 'resource_skip_count': frontend.resource_skip_count,
        'resource_skips': frontend.resource_skips, 'skipped_entity_count': frontend.skipped_count,
        'skipped_entities': frontend.skipped}, allow_incomplete_reference=True)
    metadata = {'schema_version': '1.0', 'method': METHOD, 'source_sha256': source_sha,
        'selected_layout': layout_name, 'selected_layout_native': selected, 'viewports': viewports,
        'paper_to_pixel': paper_to_pixel.tolist(), 'pixel_to_paper': np.linalg.inv(paper_to_pixel).tolist(),
        'paper_to_page': paper_to_page.tolist(), 'page_to_canvas': page_to_canvas.tolist(),
        'normalization': normalization, 'paper_content_bounds_xy': paper_bounds,
        'size_px': list(pixels.shape[::-1]), 'max_side': max_side,
        'resource_review': resource_meta, 'zero_height_mtext_policy': zero_policy,
        'audit_fixes_in_memory': len(audit.fixes), 'source_modified': False,
        'rendered_record_count': len(recorder.records), 'drawn_viewports': frontend.drawn_viewports,
        'status': 'diagnostic_layout_candidate', 'training_eligible': False,
        'annotation_approved': False, 'physical_scale_confirmed': False,
        'global_model_to_source_transform': None,
        'limitations': ['Diagnostic sheet rendering does not replace the frozen modelspace pair or resource verdict.',
            'Viewport projection is native model XY to paper coordinates and clipping; no model-to-original-PNG annotation transform is approved.',
            'Resources outside the selected viewport may not be reached; prior modelspace resource quarantine remains unchanged.',
            'Callback resource evidence is conservative: entities with unknown bounding boxes can reach a viewport callback before final clipping.',
            'Fonts, lineweights and authoring-CAD plotting can differ; observed sheet similarity is not semantic-label approval.']}
    if digest(path) != source_sha:
        raise ValueError('DXF changed during diagnostic sheet rendering')
    return pixels, metadata
