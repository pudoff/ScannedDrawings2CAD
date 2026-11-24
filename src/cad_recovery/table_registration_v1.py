"""Pixel-centre registration of a selected raster PDF page and orthogonal views.

No estimated deskew/homography is silently approximated by a table rectangle.
PDF page display coordinates, not embedded-image coordinates, retain overlays.
"""
from pathlib import Path
import math
import numpy as np

from .dataset_layout_v1 import digest, encoded
from .demo_input_v1 import _orientation_matrix
from .native_sheet_source_v1 import POLICY, prepare, _record
from .raster_stream_v2 import bounded_bytes, dimensions


def orthogonal_view(matrix, source_size):
    matrix = np.asarray(matrix, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0, 0, 1])):
        raise ValueError('Only an affine orthogonal table registration is qualified')
    for orientation in range(1, 9):
        view = np.asarray(_orientation_matrix(orientation, *source_size))
        aligned = matrix @ np.linalg.inv(view)
        if min(aligned[0, 0], aligned[1, 1]) > 0 and abs(aligned[0, 1])+abs(aligned[1, 0]) < 1e-10:
            aligned[0, 1] = aligned[1, 0] = 0.
            return dict(orientation=orientation, source_native_to_view=view.tolist(),
                native_to_working=aligned.tolist(), native_size_px=list(source_size[::-1] if orientation >= 5 else source_size))
    raise ValueError('Deskew/perspective registration is not qualified; baseline preserved')


def view_record(record, context):
    if record['size_px'] != context['source_native_size_px']: raise ValueError('Native source size changed')
    composed = np.asarray(context['source_native_to_view']) @ np.asarray(_orientation_matrix(record['orientation'], *record['raw_size_px']))
    for orientation in range(1, 9):
        if np.array_equal(composed, _orientation_matrix(orientation, *record['raw_size_px'])):
            result = {**record, 'orientation': orientation, 'size_px': context['native_size_px']}
            return result
    raise ValueError('Unsupported native orientation composition')


def pdf_geometry(source, index):
    import fitz
    import pypdfium2 as pdfium
    raw = bounded_bytes(Path(source))
    if not raw.startswith(b'%PDF-'): raise ValueError('PDF signature required')
    with fitz.open(stream=raw, filetype='pdf') as doc:
        if doc.needs_pass or not 1 <= doc.page_count <= POLICY['max_pages']:
            raise ValueError('Unencrypted raster PDF1..16 pages required')
        if type(index) is not int or not 0 <= index < doc.page_count: raise ValueError('PDF page index')
        page = doc[index]
        images = page.get_image_info()
        if not images or len(images) > 1000: raise ValueError('Selected PDF page must contain raster images')
        scales = []
        for image in images:
            a, b, c, d, _, _ = image['transform']
            sx, sy = math.hypot(a, b), math.hypot(c, d)
            if min(sx, sy) > 0: scales.append(max(image['width']/sx, image['height']/sy))
        if not scales: raise ValueError('Native PDF raster density missing')
        scale = max(scales)
        if not math.isfinite(scale) or scale <= 0: raise ValueError('Invalid PDF density')
        rect = (page.rect * fitz.Matrix(scale, scale)).irect
        dimensions(rect.width, rect.height)
        with pdfium.PdfDocument(raw) as other:
            p = other[index]
            try:
                if (p.get_rotation() != page.rotation
                        or not np.allclose(p.get_size(), [page.rect.width, page.rect.height], atol=1e-4, rtol=0)):
                    raise ValueError('PDF renderers disagree on visible page geometry')
                bbox = list(p.get_bbox())
            finally: p.close()
        return dict(schema_version='native-pdf-page-registration/1.0', page_index=index,
            page_count=doc.page_count, rotation_deg=page.rotation, visible_size_pt=[page.rect.width, page.rect.height],
            media_box_pt=list(page.mediabox), crop_box_pt=list(page.cropbox), pdfium_bbox_pt=bbox,
            size_px=[rect.width, rect.height], native_scale_pixels_per_point=scale,
            render_rect_px=list(rect), renderer=fitz.VersionBind,
            pdf_display_points_to_native=[[scale, 0, -rect.x0-.5], [0, scale, -rect.y0-.5], [0, 0, 1]])


def prepare_page(source, folder, index, expected=None):
    source, folder = Path(source), Path(folder)
    if source.suffix.lower() != '.pdf':
        if index != 0: raise ValueError('Raster page index must be zero')
        return prepare(source, folder)['pages'][0]
    import fitz
    if folder.exists(): raise FileExistsError(folder)
    geometry = pdf_geometry(source, index)
    if geometry != expected: raise ValueError('PDF page registration changed')
    before = digest(source)
    folder.mkdir(parents=True)
    surface = folder/f'page-{index+1:03d}.gray'
    scale = geometry['native_scale_pixels_per_point']
    x0, y0, x1, y1 = geometry['render_rect_px']
    width, height = geometry['size_px']
    with fitz.open(source) as doc:
        page = doc[index]
        with surface.open('xb') as stream:
            for top in range(0, height, POLICY['pdf_band_rows']):
                bottom = min(top+POLICY['pdf_band_rows'], height)
                # get_pixmap clips in the displayed frame for the pinned MuPDF
                # runtime. Full-render equivalence tests cover all four rotations.
                display = fitz.Rect((x0-1)/scale, (y0+top-1)/scale, (x1+1)/scale, (y0+bottom+1)/scale) & page.rect
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=display, colorspace=fitz.csGRAY, alpha=False)
                data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)
                x, y = x0-pix.x, y0+top-pix.y
                band = data[y:y+bottom-top, x:x+width]
                if min(x, y) < 0 or band.shape != (bottom-top, width): raise ValueError('PDF band coverage differs')
                stream.write(band.tobytes())
    if surface.stat().st_size != width*height or digest(source) != before: raise ValueError('Native PDF integrity changed')
    record = _record(surface, width, height, 1, page_index=index, pdf_geometry=geometry)
    (folder/'manifest.json').write_bytes(encoded(dict(source_sha256=before, selected_page=index, record=record)))
    return record
