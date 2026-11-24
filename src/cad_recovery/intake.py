"""Bounded local input handling and PDF vector observations (no OCR inference)."""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pypdf import PdfReader
from pypdfium2 import raw

from .schema import DrawingGraph, Issue, Line, Page, Polyline, Source, Text

PT_TO_MM = 25.4 / 72
MAX_BYTES = 50 * 1024 * 1024
MAX_PAGES = 20
MAX_PIXELS = 25_000_000


def _matrix(obj):
    matrix = obj.get_matrix()
    parent = obj.container
    while parent is not None:
        matrix = matrix.multiply(parent.get_matrix())
        parent = parent.container
    return matrix


def _display_point(point, bbox, rotation):
    left, bottom, right, top = bbox
    x, y = point[0] - left, point[1] - bottom
    w, h = right - left, top - bottom
    x, y = {0: (x, y), 90: (y, w - x), 180: (w - x, h - y), 270: (h - y, x)}[rotation]
    return x * PT_TO_MM, y * PT_TO_MM


def _bezier(start, a, b, end, steps=16):
    # A deterministic approximation, explicitly recorded as such in the graph.
    for i in range(1, steps + 1):
        t = i / steps
        yield tuple((1 - t) ** 3 * start[k] + 3 * (1 - t) ** 2 * t * a[k]
                    + 3 * (1 - t) * t * t * b[k] + t ** 3 * end[k] for k in (0, 1))


def _path_observations(obj, ref, transform):
    segments = []
    for i in range(raw.FPDFPath_CountSegments(obj)):
        segment = raw.FPDFPath_GetPathSegment(obj, i)
        x, y = ctypes.c_float(), ctypes.c_float()
        if not raw.FPDFPathSegment_GetPoint(segment, x, y):
            raise ValueError(f"Cannot read path segment: {ref}:{i}")
        segments.append((raw.FPDFPathSegment_GetType(segment), transform((x.value, y.value)),
                         bool(raw.FPDFPathSegment_GetClose(segment))))
    paths, current, closed, approximated = [], [], False, False

    def flush():
        if len(current) > 1:
            paths.append((list(current), closed))

    i = 0
    while i < len(segments):
        kind, point, close = segments[i]
        if kind == raw.FPDF_SEGMENT_MOVETO:
            flush()
            current, closed = [point], False
        elif kind == raw.FPDF_SEGMENT_LINETO and current:
            if math.dist(current[-1], point) > 1e-8:
                current.append(point)
        elif kind == raw.FPDF_SEGMENT_BEZIERTO and current:
            if i + 2 >= len(segments) or any(s[0] != kind for s in segments[i:i + 3]):
                raise ValueError(f"Incomplete Bezier path: {ref}")
            current.extend(_bezier(current[-1], point, segments[i + 1][1], segments[i + 2][1]))
            close = segments[i + 2][2]
            approximated = True
            i += 2
        else:
            raise ValueError(f"Unsupported path segment: {ref}:{i}")
        closed = closed or close
        i += 1
    flush()
    primitives = []
    for n, (points, is_closed) in enumerate(paths):
        attrs = dict(id=f"{ref}-{n}", source_ref=ref)
        if len(points) == 2 and not is_closed:
            primitives.append(Line(**attrs, start=points[0], end=points[1]))
        else:
            primitives.append(Polyline(**attrs, points=points, closed=is_closed))
    return primitives, approximated


def _pdf_graphs(source: Path, digest: str, output: Path, dpi: float):
    reader = PdfReader(source)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF is not supported")
    if not 0 < len(reader.pages) <= MAX_PAGES:
        raise ValueError(f"PDF must have 1–{MAX_PAGES} pages")
    if any(float(page.get('/UserUnit', 1)) != 1 for page in reader.pages):
        raise ValueError("PDF /UserUnit != 1 requires an explicit coordinate adapter")
    with pdfium.PdfDocument(source) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            textpage = page.get_textpage()
            try:
                width, height = page.get_size()
                if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                    raise ValueError("Invalid PDF page dimensions")
                rotation, bbox = page.get_rotation(), page.get_bbox()
                scale = min(dpi / 72, math.sqrt(MAX_PIXELS / (width * height)))
                # PDFium rounds each side up independently. A floating-point area
                # bound alone can produce a bitmap slightly over the pixel limit.
                if math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS:
                    lower, upper = 0.0, scale
                    for _ in range(48):
                        midpoint = (lower + upper) / 2
                        if math.ceil(width * midpoint) * math.ceil(height * midpoint) <= MAX_PIXELS:
                            lower = midpoint
                        else:
                            upper = midpoint
                    scale = lower
                bitmap = page.render(scale=scale)
                try:
                    bitmap.to_pil().save(output / f"page-{page_index + 1:03d}-source.png")
                finally:
                    bitmap.close()
                primitives, texts, counts = [], [], {"paths": 0, "texts": 0, "images": 0, "other": 0}
                issues, curve_count, fill_count, dashed_count, clip_count = [], 0, 0, 0, 0
                for idx, obj in enumerate(page.get_objects(textpage=textpage)):
                    ref = f"pdf-object-{idx}"
                    matrix = _matrix(obj)
                    transform = lambda xy, m=matrix: _display_point(m.on_point(*xy), bbox, rotation)
                    if obj.type == raw.FPDF_PAGEOBJ_PATH:
                        counts["paths"] += 1
                        try:
                            paths, has_curve = _path_observations(obj, ref, transform)
                            primitives.extend(paths)
                            curve_count += int(has_curve)
                            fill, stroke = ctypes.c_int(), ctypes.c_int()
                            raw.FPDFPath_GetDrawMode(obj, fill, stroke)
                            fill_count += int(fill.value != 0)
                            dashed_count += int(raw.FPDFPageObj_GetDashCount(obj) > 0)
                            clip = raw.FPDFPageObj_GetClipPath(obj)
                            if clip:
                                clip_count += int(raw.FPDFClipPath_CountPaths(clip) > 0)
                        except (ValueError, pdfium.PdfiumError) as exc:
                            issues.append(Issue(code="path_skipped", message=str(exc), entity_ids=[ref]))
                    elif obj.type == raw.FPDF_PAGEOBJ_TEXT:
                        counts["texts"] += 1
                        value = obj.extract().strip()
                        if value:
                            pos = transform((0, 0))
                            axis = transform((1, 0))
                            font_height = obj.get_font_size() * math.hypot(matrix.c, matrix.d) * PT_TO_MM
                            if font_height > 0:
                                texts.append(Text(id=ref, raw=value, position=pos, height_mm=font_height,
                                                  rotation_deg=math.degrees(math.atan2(axis[1] - pos[1], axis[0] - pos[0])),
                                                  source_ref=ref))
                    elif obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                        counts["images"] += 1
                    elif obj.type != raw.FPDF_PAGEOBJ_FORM:
                        counts["other"] += 1
                has_vector = counts["paths"] + counts["texts"] > 0
                kind = ("mixed" if has_vector else "raster") if counts["images"] else ("vector" if has_vector else "empty")
                if has_vector:
                    issues.append(Issue(code="vector_observations", message="PDF-объекты извлечены в координатах листа; классы линий и размерные связи ещё не распознаны."))
                if counts["images"]:
                    issues.append(Issue(code="raster_unprocessed", message="Растровые области сохранены в предпросмотре; CV/OCR ещё не выполнены."))
                if curve_count:
                    issues.append(Issue(code="bezier_approximated", message=f"Кривые в {curve_count} объектах приближены полилиниями (16 шагов на сегмент); точность не подтверждена."))
                if fill_count:
                    issues.append(Issue(code="fills_outline_only", message=f"Заливки {fill_count} объектов представлены только контурами."))
                if dashed_count:
                    issues.append(Issue(code="dash_styles_unresolved", message=f"Штриховые типы {dashed_count} объектов ещё не классифицированы."))
                if clip_count:
                    issues.append(Issue(code="clipping_unresolved", message=f"Области отсечения {clip_count} объектов не применены к извлечённым контурам."))
                if counts["other"]:
                    issues.append(Issue(code="unsupported_objects", message=f"Не извлечено объектов PDF: {counts['other']}."))
                if reader.pages[page_index].get('/Annots'):
                    issues.append(Issue(code="annotations_unprocessed", message="Аннотации PDF не преобразованы в CAD-объекты."))
                yield DrawingGraph(
                    source=Source(path=source.name, sha256=digest, page_index=page_index, method="pdfium-vector-observations"),
                    page=Page(width_mm=width * PT_TO_MM, height_mm=height * PT_TO_MM,
                              size_source="pdf_media", kind=kind, rotation_deg=rotation),
                    primitives=primitives, texts=texts, issues=issues,
                    provenance={"pdf_bbox_pt": list(bbox), "pdf_rotation_deg": rotation,
                                "point_to_mm": PT_TO_MM, "coordinate_origin": "visible-page-bottom-left",
                                "render_dpi": scale * 72, "object_counts": counts,
                                "bezier_steps": 16, "physical_sheet_size_verified": False},
                )
            finally:
                textpage.close()
                page.close()


def read_input(source: Path, output: Path, *, render_dpi: float = 120,
               image_dpi: float | None = None) -> list[DrawingGraph]:
    """Write source previews and return one graph per page, never consulting DXF references."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Input exceeds 50 MiB limit")
    if not math.isfinite(render_dpi) or not 36 <= render_dpi <= 600:
        raise ValueError("Render DPI must be between 36 and 600")
    if image_dpi is not None and (not math.isfinite(image_dpi) or not 36 <= image_dpi <= 2400):
        raise ValueError("Image DPI must be between 36 and 2400")
    suffix = source.suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Supported inputs: PDF, PNG, JPEG")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if suffix == ".pdf":
        return list(_pdf_graphs(source, digest, output, render_dpi))
    with Image.open(source) as original:
        if original.width * original.height > MAX_PIXELS:
            raise ValueError("Image exceeds 25 megapixel limit")
        if original.format not in {"JPEG", "PNG"}:
            raise ValueError("Image content does not match a supported format")
        metadata_dpi = original.info.get("dpi")
        orientation = original.getexif().get(274, 1)
        image = ImageOps.exif_transpose(original)
        issues = [Issue(code="raster_unprocessed", message="Растр сохранён; геометрия, OCR и размерные связи пока не распознаны.")]
        if image_dpi is not None:
            xdpi = ydpi = image_dpi
            size_source = "user"
        elif metadata_dpi and len(metadata_dpi) == 2 and all(math.isfinite(v) and 36 <= v <= 2400 for v in metadata_dpi):
            xdpi, ydpi = metadata_dpi
            if orientation in {5, 6, 7, 8}:
                xdpi, ydpi = ydpi, xdpi
            size_source = "image_dpi"
            issues.append(Issue(code="dpi_unverified", message="Размер листа рассчитан по непроверенным DPI метаданных."))
        else:
            xdpi = ydpi = 150
            size_source = "unknown"
            issues.append(Issue(code="dpi_assumed", message="DPI отсутствует. 150 DPI использовано только для координат предпросмотра; размер листа неизвестен."))
        if "A" in image.getbands() or "transparency" in image.info:
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image.convert("RGBA"))
            preview = background.convert("RGB")
        else:
            preview = image.convert("RGB")
        preview.save(output / "page-001-source.png")
        return [DrawingGraph(
            source=Source(path=source.name, sha256=digest, page_index=0, method="image-intake"),
            page=Page(width_mm=image.width / xdpi * 25.4, height_mm=image.height / ydpi * 25.4,
                      size_source=size_source, kind="raster"),
            issues=issues,
            provenance={"image_size_px": list(image.size), "dpi_xy": [xdpi, ydpi],
                        "exif_orientation_applied": orientation,
                        "pixel_to_paper_mm": [25.4 / xdpi, 0, 0, -25.4 / ydpi, 0, image.height / ydpi * 25.4]},
        )]
