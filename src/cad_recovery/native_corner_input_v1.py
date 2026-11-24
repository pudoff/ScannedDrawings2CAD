"""Authenticated original-PNG windows for short LINE/ARC boundary evidence."""
from pathlib import Path
import math
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, pin, read
from .grid_role_revision_v1 import sha
from .large_png_input_v1 import inspect_png
from .source_arrow_observation_v1 import checked_matrix, source_windows


def resize_matrix(before, after):
    sx,sy=np.asarray(after)/before
    return np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])


class NativeInput:
    def __init__(self, root, record, gray):
        self.root=Path(root); self.record=record; data=read(checked(root,record))
        for ref in data['provenance']: checked(root,ref)
        with Image.open(checked(root,data['working_gray'])) as im:
            if sha(np.array(im.convert('L'))) != sha(gray):
                raise ValueError('Native boundary context belongs to another working image')
        self.source=data['source']; self.path=checked(root,self.source)
        if self.path.suffix.lower() != '.png': raise ValueError('Original PNG required')
        header=inspect_png(self.path); self.size=header['size_px']
        if self.size != data['source_size_px']: raise ValueError('Source dimensions differ')
        self.matrix=checked_matrix(data['source_to_working'])
        if data['derivation']['kind']=='saved_prediction':
            previous=read(checked(root,data['derivation']['prediction']))
            if previous['source'] != self.source or not np.array_equal(previous['original_to_working'],self.matrix):
                raise ValueError('Prediction frame binding differs')
        elif data['derivation']['kind']=='legacy_qualification':
            previous=read(checked(root,data['derivation']['qualification']))
            if previous['source'] != self.source or not np.array_equal(previous['affine'],self.matrix):
                raise ValueError('Legacy frame binding differs')
            for ref in previous['code']+previous['provenance']: checked(root,ref)
        elif data['derivation']['kind']=='live_registered_overview':
            d=data['derivation']
            expected=(resize_matrix(d['registered_size_px'],d['working_size_px'])
                @ np.asarray(d['input_to_normalized']) @ resize_matrix(self.size,d['overview_size_px']))
            if not np.array_equal(expected,self.matrix): raise ValueError('Live affine derivation differs')
        else: raise ValueError('Unknown native frame derivation')

    def windows(self, boxes):
        return source_windows(self.path,self.source['sha256'],self.size,boxes)


def descriptor(source, size, working_gray, matrix, provenance, derivation):
    return dict(source=source,source_size_px=size,working_gray=working_gray,
        source_to_working=np.asarray(matrix).tolist(),provenance=provenance,derivation=derivation)
