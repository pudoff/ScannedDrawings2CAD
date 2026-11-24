"""Lossless JSON references for diagnostic mask arrays, without changing predictions."""
import hashlib

import numpy as np
from PIL import Image


def encode(value,folder):
    if isinstance(value,np.ndarray):
        if value.dtype!=np.uint8 or value.ndim!=2 or value.size>4194304:raise ValueError('Unexpected non-mask diagnostic array')
        digest=hashlib.sha256(value.tobytes()).hexdigest();name='diagnostic-mask-'+digest+'.png';path=folder/name
        if path.exists():
            with Image.open(path) as im:existing=np.array(im)
            if not np.array_equal(existing,value):raise ValueError('Diagnostic mask identity conflict')
        else:Image.fromarray(value).save(path)
        return {'encoding':'uint8-PNG','filename':name,'shape':list(value.shape),'pixel_sha256':digest}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:encode(v,folder) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v,folder) for v in value]
    return value
