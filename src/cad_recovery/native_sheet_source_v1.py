"""Source-resolution grayscale pages on disk; no global Pillow limit changes.

PNG is decoded once in strips. JPEG uses one bounded, full-resolution libjpeg
grayscale surface (not draft downsampling). Raster PDF is rendered in bands at
the native image density, retaining overlays. Consumers only load small crops.
"""
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from PIL import JpegImagePlugin
from .dataset_layout_v1 import digest, encoded
from .raster_stream_v2 import PNGReader, bounded_bytes, dimensions, RasterInputError
from .raster_input_v2 import jpeg_header, oriented
from .demo_input_v1 import _orientation_matrix
from .raster_trailing_normalization_v2 import png_prefix_length

POLICY = dict(max_pixels=600_000_000, max_side=32768, max_pages=16,
              max_total_pixels=600_000_000, strip_rows=64, pdf_band_rows=256,
              pixel_surface='grayscale8 on white; original bytes retained',
              jpeg_full_gray_allocation=True, max_file_bytes=50 * 1024**2)


def tiles(width, height, core=1536, halo=128):
    if (type(core) is not int or type(halo) is not int or not 64 <= core <= 1792
            or not 0 <= halo <= 128 or core + 2 * halo > 2048):
        raise ValueError('Bounded integer core/halo required')
    dimensions(width, height)
    rows = []
    for y in range(0, height, core):
        for x in range(0, width, core):
            c = [x, y, min(x + core, width), min(y + core, height)]
            rows.append(dict(index=len(rows), core=c,
                box=[max(0, x - halo), max(0, y - halo),
                     min(width, c[2] + halo), min(height, c[3] + halo)]))
    if len(rows) > 2401:
        raise ValueError('Native tile count exceeds2401')
    return rows


class NativePage:
    def __init__(self, record):
        self.record = record
        w, h = record['raw_size_px']
        self.raw = np.memmap(record['surface'], dtype=np.uint8, mode='r', shape=(h, w))
        self.orientation = record['orientation']
        self.matrix = np.asarray(_orientation_matrix(self.orientation, w, h))
        self.inverse = np.linalg.inv(self.matrix)
        self.width, self.height = record['size_px']

    def crop(self, box):
        if (len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= self.width
                or not 0 <= box[1] < box[3] <= self.height
                or (box[2] - box[0]) * (box[3] - box[1]) > 4_194_304):
            raise ValueError('Native crop must be an in-source integer box<=4MP')
        l, t, r, b = box
        p = np.array([[l, t, 1], [r - 1, t, 1], [l, b - 1, 1], [r - 1, b - 1, 1]]) @ self.inverse.T
        x, y = np.rint(p[:, :2].min(0)).astype(int)
        rr, bb = np.rint(p[:, :2].max(0)).astype(int) + 1
        return oriented(self.raw[y:bb, x:rr], self.orientation).copy()

    def samples(self, points):
        """Nearest source samples; out-of-page locations remain white."""
        p = np.asarray(points, dtype=float)
        q = np.rint(np.column_stack([p, np.ones(len(p))]) @ self.inverse.T).astype(np.int64)
        w, h = self.record['raw_size_px']
        valid = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        result = np.full(len(p), 255, np.uint8)
        result[valid] = self.raw[q[valid, 1], q[valid, 0]]
        return result

    def close(self):
        self.raw._mmap.close()


def _record(path, width, height, orientation, **extra):
    dimensions(width, height)
    return dict(surface=str(path.resolve()), surface_sha256=digest(path),
                raw_size_px=[width, height], orientation=orientation,
                size_px=[height, width] if orientation >= 5 else [width, height],
                coordinate_convention='oriented pixel centres; first centre(0,0)', **extra)


def prepare(source, folder):
    source, folder = Path(source).resolve(), Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    raw = bounded_bytes(source)
    original = dict(path=str(source), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    folder.mkdir(parents=True)
    records = []
    compatibility = None
    try:
        suffix = source.suffix.lower()
        if suffix == '.png':
            end = png_prefix_length(raw)
            path = source
            if end != len(raw):
                path = folder / 'normalized.png'
                path.write_bytes(raw[:end])
                compatibility = dict(trailing_bytes=len(raw) - end,
                    trailer_sha256=hashlib.sha256(raw[end:]).hexdigest(), normalized_sha256=digest(path))
            reader = PNGReader(path)
            w, h = reader.meta['size_px']
            surface = folder / 'page-001.gray'
            with surface.open('xb') as f:
                for _, gray in reader.iter_gray():
                    f.write(gray.tobytes())
            if not reader.validated or surface.stat().st_size != w * h:
                raise RasterInputError('input_integrity', 'Incomplete native PNG')
            records.append(_record(surface, w, h, reader.meta['orientation'],
                                  decoder='PNG strip64; one complete CRC/zlib/SHA pass', full_bitmap_in_RAM=False))
        elif suffix in ('.jpg', '.jpeg'):
            meta = jpeg_header(source, raw)
            w, h = meta['size_px']
            surface = folder / 'page-001.gray'
            with JpegImagePlugin.JpegImageFile(io.BytesIO(raw)) as image:
                image.draft('L', (w, h))
                if image.size != (w, h) or image.mode != 'L':
                    raise RasterInputError('input_jpeg_native', 'Native grayscale libjpeg decode required')
                image.load()
                with surface.open('xb') as f:
                    for y in range(0, h, POLICY['strip_rows']):
                        f.write(image.crop((0, y, w, min(y + POLICY['strip_rows'], h))).tobytes())
            records.append(_record(surface, w, h, meta['orientation'],
                decoder='libjpeg full-resolution grayscale; no resize', full_bitmap_in_RAM=True))
        elif suffix == '.pdf':
            import fitz
            if not raw.startswith(b'%PDF-'):
                raise RasterInputError('input_signature', 'PDF signature required')
            with fitz.open(stream=raw, filetype='pdf') as document:
                if document.needs_pass or not 1 <= document.page_count <= POLICY['max_pages']:
                    raise RasterInputError('input_pdf_pages', 'Unencrypted PDF1..16pages required')
                total = 0
                for index, page in enumerate(document):
                    images = page.get_image_info()
                    if not images or len(images) > 1000:
                        raise RasterInputError('input_pdf_raster', 'Raster PDF required; vector branch remains separate')
                    scales = []
                    for image in images:
                        a, b, c, d, _, _ = image['transform']
                        sx, sy = math.hypot(a, b), math.hypot(c, d)
                        if sx > 0 and sy > 0:
                            scales.append(max(image['width'] / sx, image['height'] / sy))
                    if not scales:
                        raise RasterInputError('input_pdf_raster', 'Finite placed raster required')
                    scale = max(scales)
                    if not math.isfinite(scale) or scale <= 0:
                        raise RasterInputError('input_pdf_raster', 'Invalid raster density')
                    matrix = fitz.Matrix(scale, scale)
                    rect = (page.rect * matrix).irect
                    w, h = rect.width, rect.height
                    dimensions(w, h)
                    total += w * h
                    if total > POLICY['max_total_pixels']:
                        raise RasterInputError('input_pdf_pixels', 'PDF total native raster exceeds600MP')
                    surface = folder / f'page-{index + 1:03d}.gray'
                    with surface.open('xb') as f:
                        for top in range(0, h, POLICY['pdf_band_rows']):
                            bottom = min(top + POLICY['pdf_band_rows'], h)
                            # A margin absorbs renderer floor/ceil rounding; crop by pixmap origin.
                            clip = fitz.Rect((rect.x0 - 1) / scale, (rect.y0 + top - 1) / scale,
                                             (rect.x1 + 1) / scale, (rect.y0 + bottom + 1) / scale) & page.rect
                            pix = page.get_pixmap(matrix=matrix, clip=clip, colorspace=fitz.csGRAY, alpha=False)
                            data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)
                            x, y = rect.x0 - pix.x, rect.y0 + top - pix.y
                            band = data[y:y + bottom - top, x:x + w]
                            if x < 0 or y < 0 or band.shape != (bottom - top, w):
                                raise RasterInputError('input_pdf_coverage', 'PDF band extent differs')
                            f.write(band.tobytes())
                    records.append(_record(surface, w, h, 1, page_index=index,
                        pdf_points_to_pixel=[[scale, 0, -rect.x0 - .5], [0, scale, -rect.y0 - .5], [0, 0, 1]],
                        native_scale_pixels_per_point=scale, decoder='MuPDF native-density band render; overlays included',
                        full_bitmap_in_RAM=False, renderer_may_cache_embedded_images=True))
        else:
            raise RasterInputError('input_extension', 'PNG/JPEG/raster-PDF required')
        if digest(source) != original['sha256']:
            raise RasterInputError('input_integrity', 'Original changed during preparation')
        report = dict(schema_version='native-sheet-source/1.0', source=original, policy=POLICY,
                      pages=records, compatibility_normalization=compatibility,
                      source_unchanged=True, resize_applied=False, physical_geometry_approved=False)
        (folder / 'manifest.json').write_bytes(encoded(report))
        return report
    except Exception as error:
        (folder / 'failure.json').write_bytes(encoded(dict(error=str(error), accepted=False)))
        raise
