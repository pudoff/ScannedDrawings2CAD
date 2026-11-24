"""Bounded scanline decoding of static, 8-bit RGB, non-interlaced PNG files.

Full-size images are never passed to Pillow. Each small strip is decoded as a
synthetic PNG preceded by the preceding decoded row, preserving PNG filters at
strip boundaries. No decoder safety setting is changed. Yielded strips remain
provisional until the iterator is exhausted and ``validated`` becomes true.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_SOURCE_PIXELS = 300_000_000
MAX_SIDE = 32768
MAX_CHUNKS = 10000
MAX_STRIP_ROWS = 256
READ_BYTES = 64 * 1024
SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _header(data: bytes) -> tuple[int, int]:
    if len(data) != 13:
        raise ValueError("PNG IHDR must have 13 bytes")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
    if not 0 < width <= MAX_SIDE or not 0 < height <= MAX_SIDE or width * height > MAX_SOURCE_PIXELS:
        raise ValueError("PNG source dimensions exceed bounded-reader limits")
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("Bounded PNG reader supports only 8-bit RGB, non-interlaced static PNG")
    return width, height


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF))


def _decode_strip(filtered: bytearray, previous: bytes, width: int, rows: int) -> np.ndarray:
    stride = width * 3 + 1
    # Pillow also validates filter bytes; check explicitly for a useful error.
    if any(filtered[offset] > 4 for offset in range(0, len(filtered), stride)):
        raise ValueError("PNG scanline has an invalid filter type")
    compressor = zlib.compressobj(level=1)
    compressed = compressor.compress(b"\x00" + previous)
    compressed += compressor.compress(filtered)
    compressed += compressor.flush()
    png = (SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, rows + 1, 8, 2, 0, 0, 0)) +
           _chunk(b"IDAT", compressed) + _chunk(b"IEND", b""))
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        # Own memory: callers may retain a strip without depending on the next.
        result = np.array(image, dtype=np.uint8)[1:].copy()
    if result.shape != (rows, width, 3):
        raise ValueError("Unexpected bounded PNG strip dimensions")
    return result


class PNGStripReader:
    """Read PNG strips without allocating a full-resolution raster.

    Source bytes, dimensions, chunk count and strip height are bounded. Every
    chunk CRC, the single zlib stream, scanline count, IEND and final source hash
    must validate before ``validated`` is true. Partial consumption is not full
    validation. APNG, transparency and EXIF orientation are explicitly unsupported.
    Ancillary metadata is skipped without decompressing embedded payloads.
    """

    def __init__(self, path: Path, expected_sha256: str | None = None, strip_rows: int = 128):
        self.path = Path(path)
        if isinstance(strip_rows, bool) or not isinstance(strip_rows, int) or not 1 <= strip_rows <= MAX_STRIP_ROWS:
            raise ValueError(f"strip_rows must be an integer in 1..{MAX_STRIP_ROWS}")
        self.strip_rows = strip_rows
        self.source_size_bytes = self.path.stat().st_size
        if not 0 < self.source_size_bytes <= MAX_FILE_BYTES:
            raise ValueError("PNG source exceeds the 50 MiB file limit")
        with self.path.open("rb") as stream:
            self.source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and self.source_sha256 != expected_sha256:
            raise ValueError("PNG source_sha256 differs from expected source")
        with self.path.open("rb") as stream:
            header = stream.read(33)
        if len(header) != 33 or header[:8] != SIGNATURE or header[8:16] != b"\x00\x00\x00\rIHDR":
            raise ValueError("PNG must begin with its signature and IHDR")
        if (zlib.crc32(header[12:29]) & 0xFFFFFFFF) != struct.unpack(">I", header[29:33])[0]:
            raise ValueError("PNG IHDR CRC mismatch")
        self.width, self.height = _header(header[16:29])
        self.metadata = {
            "format": "PNG", "width_px": self.width, "height_px": self.height,
            "bit_depth": 8, "color_type": 2, "interlace_method": 0,
            "source_sha256": self.source_sha256, "source_size_bytes": self.source_size_bytes,
            "decoder": "bounded_png_strips_v1", "strip_rows": strip_rows,
            "decoder_limits_overridden": False,
            "physical_pixel_dimensions": None,
            "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_source_pixels": MAX_SOURCE_PIXELS,
                       "max_side": MAX_SIDE, "max_chunks": MAX_CHUNKS, "read_bytes": READ_BYTES},
        }
        self.validated = False
        self.decoded_rows = 0
        self.bytes_read = 0
        self.chunk_count = 0
        self._started = False

    def _compressed_pieces(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            def read(count):
                data = stream.read(count)
                digest.update(data)
                self.bytes_read += len(data)
                if self.bytes_read > MAX_FILE_BYTES:
                    raise ValueError("PNG source exceeds the file limit while reading")
                if len(data) != count:
                    raise ValueError("Truncated PNG chunk or data")
                return data

            if read(8) != SIGNATURE:
                raise ValueError("PNG signature changed during reading")
            idat_state = 0  # before, inside, after the consecutive IDAT group
            palette_seen = False
            physical_seen = False
            while True:
                length, kind = struct.unpack(">I4s", read(8))
                self.chunk_count += 1
                if self.chunk_count > MAX_CHUNKS:
                    raise ValueError("PNG chunk count exceeds the limit")
                if length > MAX_FILE_BYTES or self.bytes_read + length + 4 > self.source_size_bytes:
                    raise ValueError("PNG chunk length exceeds the source/file limit")
                if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in kind) or not 65 <= kind[2] <= 90:
                    raise ValueError("Invalid PNG chunk type")
                if self.chunk_count == 1 and kind != b"IHDR":
                    raise ValueError("PNG IHDR must be the first chunk")
                if kind == b"IHDR" and (self.chunk_count != 1 or length != 13):
                    raise ValueError("Duplicate or invalid PNG IHDR")
                if kind in {b"acTL", b"fcTL", b"fdAT", b"tRNS", b"eXIf"}:
                    raise ValueError(f"Unsupported PNG chunk: {kind.decode('ascii')}")
                if kind not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"} and kind[0] < 97:
                    raise ValueError(f"Unknown critical PNG chunk: {kind.decode('ascii')}")
                if kind == b"PLTE":
                    if palette_seen or idat_state != 0 or not 0 < length <= 768 or length % 3:
                        raise ValueError("Invalid or misplaced PNG PLTE")
                    palette_seen = True
                if kind == b"pHYs":
                    if physical_seen or idat_state != 0 or length != 9:
                        raise ValueError("Invalid, duplicate or misplaced PNG pHYs")
                    physical_seen = True
                if kind == b"IDAT":
                    if idat_state == 2:
                        raise ValueError("PNG IDAT chunks must be consecutive")
                    idat_state = 1
                elif idat_state == 1:
                    idat_state = 2
                if kind == b"IEND" and (length != 0 or idat_state == 0):
                    raise ValueError("Invalid PNG IEND or missing IDAT")
                crc = zlib.crc32(kind)
                remaining = length
                payload = b""
                while remaining:
                    data = read(min(remaining, READ_BYTES))
                    crc = zlib.crc32(data, crc)
                    remaining -= len(data)
                    if kind in {b"IHDR", b"pHYs"}:
                        payload += data
                    elif kind == b"IDAT":
                        yield data
                if (crc & 0xFFFFFFFF) != struct.unpack(">I", read(4))[0]:
                    raise ValueError(f"PNG {kind.decode('ascii')} CRC mismatch")
                if kind == b"IHDR" and _header(payload) != (self.width, self.height):
                    raise ValueError("PNG dimensions changed during reading")
                if kind == b"pHYs":
                    x, y, unit = struct.unpack(">IIB", payload)
                    if unit not in (0, 1):
                        raise ValueError("Invalid PNG pHYs unit")
                    self.metadata["physical_pixel_dimensions"] = {
                        "pixels_per_unit_x": x, "pixels_per_unit_y": y,
                        "unit": "meter" if unit == 1 else "unknown",
                        "physical_scale_confirmed": False,
                    }
                if kind == b"IEND":
                    if stream.read(1):
                        raise ValueError("Unexpected data after PNG IEND")
                    break
            if self.bytes_read != self.source_size_bytes or digest.hexdigest() != self.source_sha256:
                raise ValueError("PNG source changed during streaming")

    def iter_strips(self):
        """Yield ``(source_y, RGB uint8 pixels)``; exhaust to validate the source."""
        if self._started:
            raise ValueError("PNGStripReader supports one iteration only")
        self._started = True
        stride = self.width * 3 + 1
        expected_bytes = stride * self.height
        decoded_bytes = 0
        buffer = bytearray()
        previous = bytes(self.width * 3)
        inflater = zlib.decompressobj()
        try:
            for piece in self._compressed_pieces():
                pending = piece
                if inflater.eof:
                    raise ValueError("Unexpected compressed data after PNG zlib stream")
                while pending:
                    rows = min(self.strip_rows, self.height - self.decoded_rows)
                    target = rows * stride
                    room = max(1, target - len(buffer))
                    decoded = inflater.decompress(pending, room)
                    decoded_bytes += len(decoded)
                    if decoded_bytes > expected_bytes:
                        raise ValueError("PNG zlib output exceeds declared scanline count")
                    buffer.extend(decoded)
                    pending = inflater.unconsumed_tail
                    if target and len(buffer) == target:
                        pixels = _decode_strip(buffer, previous, self.width, rows)
                        previous = pixels[-1].tobytes()
                        start = self.decoded_rows
                        self.decoded_rows += rows
                        buffer.clear()
                        yield start, pixels
                    if inflater.eof:
                        if inflater.unused_data or pending:
                            raise ValueError("Extra or concatenated PNG zlib streams")
                        break
        except zlib.error as error:
            raise ValueError(f"Invalid PNG zlib stream: {error}") from error
        if not inflater.eof or decoded_bytes != expected_bytes or buffer or self.decoded_rows != self.height:
            raise ValueError("Truncated PNG zlib stream or incorrect scanline count")
        self.validated = True
        self.metadata.update(decoded_rows=self.decoded_rows, streamed_bytes=self.bytes_read,
                             chunk_count=self.chunk_count, integrity_validated=True)
