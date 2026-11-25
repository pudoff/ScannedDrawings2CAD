"""Versioned reference renderer: bounded zero-height MTEXT and explicit resources.

The original renderer is retained. Its placement, color, masking and hatch
policies are reused; only the documented MTEXT/resource branches are new.
"""
from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import random
import warnings

import ezdxf
import numpy as np
import pymupdf
from ezdxf.addons.drawing import RenderContext, config, layout
from ezdxf.addons.drawing.pymupdf import PyMuPdfRenderBackend, get_coordinate_output_space
from ezdxf.units import unit_name

from .dxf_reference import (MAX_FILE_BYTES, MAX_ENTITIES, MAX_MESSAGES, MAX_RECORDS,
    HATCH_JITTER_SEED, _HATCH_RANDOM_LOCK, _digest, _LogCapture, _Recorder,
    _Frontend as OriginalFrontend, _matrix_xy, normalize_reference)
from .mtext_zero_height import install_zero_height_adapter
from .reference_resources import ResourceAwareFrontendMixin


class _Frontend(ResourceAwareFrontendMixin, OriginalFrontend):
    pass


def render_dxf_reference(path: Path, max_side: int = 1600, *, allow_incomplete_reference: bool = False,
                         resource_root: Path | None = None) -> tuple[np.ndarray, dict]:
    """Render modelspace and return grayscale pixels plus reversible WCS XY mapping.

    Mapping uses native CAD coordinates irrespective of INSUNITS. Units and
    renderer substitutions are evidence only; they do not validate dimensions.
    Rendering is planar XY; Z is projected by the drawing frontend.

    The ezdxf hatch renderer uses Python's global random generator. Drawing is
    serialized among calls to this function and temporarily seeds/restores that
    generator, including on failure. Use this API sequentially (as the CLI does)
    when other application threads also use the global generator.
    """
    path = Path(path).resolve()
    if not isinstance(allow_incomplete_reference, bool):
        raise ValueError("allow_incomplete_reference must be boolean")
    resource_root = Path(resource_root).resolve() if resource_root is not None else path.parent
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 320 <= max_side <= 2400:
        raise ValueError("max_side must be an integer between 320 and 2400")
    if path.suffix.lower() != ".dxf" or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Reference requires a DXF no larger than 50 MiB")
    source_hash = _digest(path)
    logger = logging.getLogger("ezdxf")
    logs = _LogCapture()
    logger.addHandler(logs)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            document = ezdxf.readfile(path)
            if len(document.entitydb) > MAX_ENTITIES:
                raise ValueError("DXF reference exceeds the entity limit")
            original_counts = dict(Counter(entity.dxftype() for entity in document.modelspace()))
            audit = document.audit()
            if audit.errors:
                raise ValueError("DXF contains unresolved audit errors")
            zero_height_policy = install_zero_height_adapter(document)
            context = RenderContext(document)
            font_styles = []
            for style in document.styles:
                face = context.fonts.get(style.dxf.name.casefold())
                requested = style.dxf.get("font", "")
                resolved = face.filename if face else ""
                font_styles.append({"style": style.dxf.name, "requested_font": requested,
                                    "resolved_font": resolved,
                                    "bigfont": style.dxf.get("bigfont", ""),
                                    "substituted_or_default": not requested or not resolved or
                                    Path(requested).name.casefold() != Path(resolved).name.casefold()})
            recorder = _Recorder()
            options = config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                                           color_policy=config.ColorPolicy.COLOR,
                                           hatching_timeout=5.0)
            frontend = _Frontend(context, recorder, config=options)
            frontend.configure_resources(path, resource_root, allow_incomplete_reference=allow_incomplete_reference)
            # ezdxf jitters hatch origins to avoid corner-intersection artifacts.
            # An uncontrolled jitter can change borderline SIFT decisions. Keep
            # the supported jitter algorithm, but make each document reproducible
            # and avoid consuming the sequential caller's random state.
            with _HATCH_RANDOM_LOCK:
                caller_random_state = random.getstate()
                try:
                    random.seed(HATCH_JITTER_SEED)
                    frontend.draw_layout(document.modelspace(), finalize=True)
                finally:
                    random.setstate(caller_random_state)
            player = recorder.player()
            bounds = player.bbox()
            if not bounds.has_data or min(bounds.size) <= 0 or not np.isfinite(tuple(bounds.extmin) + tuple(bounds.extmax)).all():
                raise ValueError("DXF reference has empty, degenerate or nonfinite drawing bounds")
            cad_bounds = [*bounds.extmin, *bounds.extmax]
            handles = sorted({record.handle for record in recorder.records if record.handle})
            layers = sorted({properties.layer for _, properties in player.recordings()})
            # This is the same documented layout placement used by PyMuPdfBackend,
            # but replay is explicit so the exact transform is retained.
            page = layout.Page(297, 297, margins=layout.Margins.all(3))
            settings = layout.Settings(output_coordinate_space=get_coordinate_output_space(page))
            placement = layout.Layout(bounds, flip_y=True).get_placement_matrix(
                page, settings=settings, top_origin=True)
            cad_to_page = _matrix_xy(placement)
            player.transform(placement)
            backend = PyMuPdfRenderBackend(page, settings)
            try:
                player.replay(backend)
                page_size = [backend.page_width_in_pt, backend.page_height_in_pt]
                raster_scale = (max_side * 1.5) / max(page_size)
                pixmap = backend.page.get_pixmap(matrix=pymupdf.Matrix(raster_scale, raster_scale),
                                                colorspace=pymupdf.csGRAY, alpha=False)
                canvas = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy()
                # PyMuPDF origin is a pixel edge; array indices denote centers.
                page_to_canvas = np.array([[raster_scale, 0., -pixmap.x-.5],
                                           [0., raster_scale, -pixmap.y-.5], [0., 0., 1.]])
                canvas_size = [pixmap.width, pixmap.height]
            finally:
                backend.doc.close()
            result, normalization = normalize_reference(canvas, max_side)
            transform = np.array(normalization["input_to_normalized"]) @ page_to_canvas @ cad_to_page
            if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-16:
                raise ValueError("DXF reference has an invalid coordinate transform")
            metadata = {
                "method": "ezdxf-recorder-explicit-placement-pymupdf-v4-translation-only-mtext",
                "zero_height_mtext_policy": zero_height_policy,
                "reference_complete": frontend.reference_complete,
                "reference_usage": "comparison_candidate" if frontend.reference_complete else "diagnostic_only_incomplete_reference",
                "resource_observations": frontend.resource_observations,
                "incomplete_reference_explicitly_allowed": allow_incomplete_reference,
                "color_policy": "resolved white stays white, all other foreground colors become black; alpha preserved",
                "hatch_jitter_seed": HATCH_JITTER_SEED,
                "hatch_jitter_meaning": "deterministic renderer perturbation; reproducibility is not geometric accuracy",
                "random_state_policy": "save/seed/restore Python global RNG around drawing; serialized renderer calls; sequential API",
                "white_record_count": sum(recorder.properties[record.property_hash].color[:7] == "#ffffff"
                                          for record in recorder.records),
                "source_sha256": source_hash, "source_modified": False,
                "size_px": list(result.shape[::-1]), "max_side": max_side,
                "cad_to_pixel": transform.tolist(), "pixel_to_cad": np.linalg.inv(transform).tolist(),
                "coordinate_convention": "DXF WCS XY to pixel centers; image X right and Y down",
                "units": int(document.units), "units_name": unit_name(document.units),
                "physical_scale_confirmed": False, "cad_semantics_verified": False,
                "modelspace_entities_before_audit": original_counts,
                "rendered_record_count": len(recorder.records),
                "rendered_handles": handles, "rendered_layers": layers,
                "handle_scope": "drawing-backend handles; composites may refer to their parent",
                "cad_content_bounds_xy": cad_bounds,
                "canvas_size_px": canvas_size, "page_size_pt": page_size,
                "cad_to_page": cad_to_page.tolist(), "page_to_canvas": page_to_canvas.tolist(),
                "normalization": normalization,
                "font_styles": font_styles,
                "audit": {"error_count": 0, "fix_count": len(audit.fixes), "fixes_saved": False,
                          "fixes": [{"code": int(fix.code), "message": fix.message} for fix in audit.fixes[:MAX_MESSAGES]]},
                "skipped_entity_count": frontend.skipped_count, "skipped_entities": frontend.skipped,
                "frontend_message_count": frontend.message_count, "frontend_messages": frontend.messages,
                "log_warning_count": logs.total, "log_warnings": logs.messages,
                "python_warning_count": len(caught), "python_warnings": [str(item.message) for item in caught[:MAX_MESSAGES]],
                "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_entities": MAX_ENTITIES,
                           "max_drawing_records": MAX_RECORDS, "hatching_timeout_seconds": 5.0},
                "limitations": ["Only pure translation is supported for transformed zero-base-height MTEXT; native text and height attributes are preserved.",
                                "Missing external images and opaque OLE content are explicit diagnostic placeholders, never a complete reference.",
                                "Zero-width MTEXT retains the existing renderer layout estimate; authoring-CAD layout accuracy is not asserted.",
                                "Modelspace planar XY rendering, not a paperspace viewport mapping.",
                                "Fonts, proxy graphics, lineweights and hatch rendering can differ from the authoring CAD.",
                                "Rendered handles indicate drawing provenance, not accepted semantic labels.",
                                "Coordinate invertibility does not establish physical scale or PNG/DXF correspondence."],
            }
    finally:
        logger.removeHandler(logs)
    if _digest(path) != source_hash:
        raise ValueError("DXF source changed during reference rendering")
    return result, metadata
