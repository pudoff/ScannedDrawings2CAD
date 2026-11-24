"""Validated public window API over the immutable qualified strip kernel."""
from .raster_input_v2 import native_windows as _read_windows
from .raster_stream_v2 import RasterInputError


def native_windows(source,boxes,expected_sha=None):
    # Validate every area before any allocation. A negative malformed box must
    # never cancel a positive large box in the aggregate memory calculation.
    if not isinstance(boxes,(list,tuple)) or not 1<=len(boxes)<=64:
        raise RasterInputError('input_windows','Window count1..64 required')
    total=0
    for box in boxes:
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(n) is not int for n in box):
            raise RasterInputError('input_windows','Integer boxes required')
        l,t,r,b=box
        if not 0<=l<r or not 0<=t<b:raise RasterInputError('input_windows','Window must have positive in-source extent')
        total+=(r-l)*(b-t)
        if total>16_000_000:raise RasterInputError('input_windows','Total window area exceeds16MP')
    return _read_windows(source,boxes,expected_sha)
