"""Bounded DXF reference rendering with the actual CAD-to-pixel transform.

This is a reference for dataset review, not confirmation of CAD semantics or
physical scale. All placement, Y reversal, crop and pixel-center resize steps
are recorded. The source document is never saved.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import logging
from pathlib import Path
import random
import threading
import warnings

import cv2
import ezdxf
import numpy as np
from ezdxf.addons.drawing import Frontend, RenderContext, config, layout
from ezdxf.addons.drawing.pymupdf import PyMuPdfRenderBackend, get_coordinate_output_space
from ezdxf.addons.drawing.recorder import Recorder
from ezdxf.units import unit_name
import pymupdf

MAX_FILE_BYTES = 50 * 1024**2
MAX_ENTITIES = 200_000
MAX_RECORDS = 250_000
MAX_MESSAGES = 100
HATCH_JITTER_SEED = 0
_HATCH_RANDOM_LOCK = threading.Lock()


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _Recorder(Recorder):
    def store(self, record, properties):
        if len(self.records) >= MAX_RECORDS:
            raise ValueError("DXF reference exceeds the drawing-record limit")
        # Apply monochrome only after the frontend resolves background masks.
        # ColorPolicy.BLACK would also turn WIPEOUT/MTEXT white backgrounds into
        # opaque black polygons, hiding text and changing reference geometry.
        color = properties.color
        mapped = "#ffffff" if color[:7].lower() == "#ffffff" else "#000000"
        properties = properties._replace(color=mapped + color[7:])
        super().store(record, properties)


class _Frontend(Frontend):
    def __init__(self, *args, **kwargs):
        self.skipped = []
        self.skipped_count = 0
        self.messages = []
        self.message_count = 0
        super().__init__(*args, **kwargs)

    def skip_entity(self, entity, msg):
        self.skipped_count += 1
        if len(self.skipped) < MAX_MESSAGES:
            self.skipped.append({"handle": entity.dxf.get("handle"),
                                 "type": entity.dxftype(), "reason": msg})

    def log_message(self, message):
        self.message_count += 1
        if len(self.messages) < MAX_MESSAGES:
            self.messages.append(str(message))

    def draw_simple_mtext(self, mtext, properties):
        # ezdxf's simple text path ignores background fill. Reuse its complete
        # layout renderer for masks even when the string has no inline codes.
        if mtext.dxf.get("bg_fill", 0):
            self.draw_complex_mtext(mtext, properties)
        else:
            super().draw_simple_mtext(mtext, properties)


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []
        self.total = 0

    def emit(self, record):
        self.total += 1
        if len(self.messages) < MAX_MESSAGES:
            self.messages.append(record.getMessage())


def normalize_reference(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, dict]:
    """Crop visible ink and resize, recording input-pixel to output-pixel centers."""
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 320 <= max_side <= 2400:
        raise ValueError("max_side must be an integer between 320 and 2400")
    if gray.dtype != np.uint8 or gray.ndim != 2 or max(gray.shape) > 4096:
        raise ValueError("Reference normalization requires a bounded uint8 grayscale image")
    ys, xs = np.where(gray < 210)
    if len(xs) < 100:
        raise ValueError("Empty or insufficient drawing content")
    left, top = max(0, int(xs.min()) - 3), max(0, int(ys.min()) - 3)
    right, bottom = min(gray.shape[1], int(xs.max()) + 4), min(gray.shape[0], int(ys.max()) + 4)
    width, height = right-left, bottom-top
    scale = max_side / max(width, height)
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    sx, sy = size[0]/width, size[1]/height
    # OpenCV resize maps center coordinates by (x+.5)*scale-.5.
    matrix = np.array([[sx, 0., sx*(.5-left)-.5],
                       [0., sy, sy*(.5-top)-.5], [0., 0., 1.]])
    result = cv2.resize(gray[top:bottom, left:right], size,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return result, {"input_size_px": list(gray.shape[::-1]), "size_px": list(size),
                    "crop_box_px": [left, top, right, bottom],
                    "input_to_normalized": matrix.tolist(),
                    "normalized_to_input": np.linalg.inv(matrix).tolist(),
                    "threshold": 210, "crop_padding_px": 3,
                    "coordinate_convention": "integer coordinates are pixel centers"}


def _matrix_xy(matrix):
    # Use the library transform itself, avoiding row/column-major assumptions.
    origin = np.array(tuple(matrix.transform((0, 0, 0)))[:2])
    x = np.array(tuple(matrix.transform((1, 0, 0)))[:2]) - origin
    y = np.array(tuple(matrix.transform((0, 1, 0)))[:2]) - origin
    return np.array([[x[0], y[0], origin[0]], [x[1], y[1], origin[1]], [0., 0., 1.]])


def render_dxf_reference(path: Path, max_side: int = 1600) -> tuple[np.ndarray, dict]:
    """Render modelspace and return grayscale pixels plus reversible WCS XY mapping.

    Mapping uses native CAD coordinates irrespective of INSUNITS. Units and
    renderer substitutions are evidence only; they do not validate dimensions.
    Rendering is planar XY; Z is projected by the drawing frontend.

    The ezdxf hatch renderer uses Python's global random generator. Drawing is
    serialized among calls to this function and temporarily seeds/restores that
    generator, including on failure. Use this API sequentially (as the CLI does)
    when other application threads also use the global generator.
    """
    path = Path(path)
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
                "method": "ezdxf-recorder-explicit-placement-pymupdf-v3",
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
                "limitations": ["Modelspace planar XY rendering, not a paperspace viewport mapping.",
                                "Fonts, proxy graphics, lineweights and hatch rendering can differ from the authoring CAD.",
                                "Rendered handles indicate drawing provenance, not accepted semantic labels.",
                                "Coordinate invertibility does not establish physical scale or PNG/DXF correspondence."],
            }
    finally:
        logger.removeHandler(logs)
    if _digest(path) != source_hash:
        raise ValueError("DXF source changed during reference rendering")
    return result, metadata
