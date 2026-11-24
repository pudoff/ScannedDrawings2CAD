"""Bind cached native page masks to the actual AREA/BOX producer of one view."""
import cv2
import numpy as np
from PIL import Image
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


def bind(packet,gray,native,proof,matrix):
    before=object_sha(packet);method=proof.get('resize_method')
    if method not in ('CV_AREA','PIL_BOX') or proof.get('GT_used') is not False:
        raise ValueError('Explicit exact source resize provenance required')
    if proof['input_sha256']!=sha(gray) or proof['source_sha256']!=sha(native) or proof['matrix']!=matrix:
        raise ValueError('View/source/matrix proof differs')
    if gray.shape==native.shape:expected=native
    elif method=='CV_AREA':expected=cv2.resize(native,gray.shape[::-1],interpolation=cv2.INTER_AREA)
    else:expected=np.array(Image.fromarray(native).resize(gray.shape[::-1],Image.Resampling.BOX))
    if not np.array_equal(gray,expected):raise ValueError('Resize provenance is not exact')
    for record in packet['masks'].values():
        if record['source_gray_sha256']!=sha(native):raise ValueError('Page masks use another source')
    bound={**packet,'masks':{k:{**record,'resize_method':method,'view_resize_binding':proof}
        for k,record in packet['masks'].items()},'view_resize_binding':proof}
    if before!=object_sha(packet):raise ValueError('Page view binding mutated the cache')
    return bound
