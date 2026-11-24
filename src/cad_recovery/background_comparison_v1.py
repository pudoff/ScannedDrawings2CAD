"""Bounded offline background ablations, deliberately separate from service profiles."""
from __future__ import annotations

import cv2
import numpy as np

VARIANTS = ('baseline', 'otsu', 'gaussian_otsu', 'bilateral_otsu',
            'closing_adaptive', 'sauvola', 'wolf', 'rolling_adaptive')


def check(gray):
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 3:
        raise ValueError('Expected uint8 grayscale >=3x3')
    if gray.size > 1_000_000:
        raise ValueError('Offline experiment limited to one megapixel')


def adaptive(gray):
    shortest = min(gray.shape)
    window = min(101, max(15, shortest // 30)) | 1
    window = min(window, shortest if shortest % 2 else shortest - 1)
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,window,12)


def local_threshold(gray, kind, window=45):
    """Float64 Sauvola/Wolf with replicated boundaries and explicit flat fallback.

    Mathematical thresholds are not rounded to uint8, unlike ximgproc.
    """
    if kind not in ('sauvola','wolf') or window < 3 or window % 2 != 1:
        raise ValueError('Invalid local threshold parameters')
    f=gray.astype(np.float64)
    mean=cv2.boxFilter(f,-1,(window,window),borderType=cv2.BORDER_REPLICATE)
    square=cv2.boxFilter(f*f,-1,(window,window),borderType=cv2.BORDER_REPLICATE)
    std=np.sqrt(np.maximum(0,square-mean*mean))
    if kind=='sauvola':
        threshold=mean*(1+.2*(std/128-1))
    else:
        maximum=float(std.max())
        if maximum < 1e-9:
            return np.full(gray.shape,255,dtype=np.uint8)
        threshold=mean-.5*(mean-f.min())*(1-std/maximum)
    return np.where(f>threshold,255,0).astype(np.uint8)


def rolling_background(gray, radius=27):
    """Spherical nonflat minimum of inverted image, with replicated boundaries.

    Ball height penalty r-sqrt(r²-d²); intensity and spatial radii both r.
    Transparent NumPy reference, not a bitwise clone of a third-party library.
    """
    if not 1 <= radius <= 32:
        raise ValueError('Bounded radius must be in 1..32')
    inverted=255-gray.astype(np.float32)
    padded=np.pad(inverted,radius,mode='edge')
    result=np.full(gray.shape,np.inf,dtype=np.float32)
    height,width=gray.shape
    for y in range(-radius,radius+1):
        for x in range(-radius,radius+1):
            squared=x*x+y*y
            if squared<=radius*radius:
                penalty=radius-np.sqrt(radius*radius-squared)
                patch=padded[radius+y:radius+y+height,radius+x:radius+x+width]
                np.minimum(result,patch+penalty,out=result)
    return np.clip(gray.astype(np.float32)+result,0,255).round().astype(np.uint8)


def apply_variant(gray, variant):
    check(gray)
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    prepared=gray.copy()
    if variant=='gaussian_otsu':
        prepared=cv2.GaussianBlur(gray,(3,3),0)
    elif variant=='bilateral_otsu':
        # OpenCV 5 optimized uint8 path can turn constant 255 into 254/255,
        # which Otsu interprets as ink. Float filtering + explicit rounding
        # preserves the constant-field invariant in train qualification.
        prepared=np.clip(cv2.bilateralFilter(gray.astype(np.float32),5,25,3),0,255).round().astype(np.uint8)
    elif variant=='closing_adaptive':
        background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55)),borderType=cv2.BORDER_REPLICATE)
        prepared=np.clip(255-(background.astype(np.int16)-gray),0,255).astype(np.uint8)
    elif variant=='rolling_adaptive':
        prepared=rolling_background(gray)
    if variant in ('sauvola','wolf'):
        binary=local_threshold(prepared,variant)
    elif variant.endswith('otsu') or variant=='otsu':
        binary=cv2.threshold(prepared,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    else:
        binary=adaptive(prepared)
    return prepared,binary==0


def synthetic_training():
    """Procedural train-only controls; no customer/test pixels or targets."""
    ink=np.zeros((160,240),dtype=np.uint8)
    cv2.rectangle(ink,(20,20),(100,100),255,3)
    cv2.circle(ink,(60,60),16,255,3)
    cv2.line(ink,(120,30),(210,30),255,1)
    cv2.line(ink,(120,30),(129,25),255,1)
    cv2.line(ink,(120,30),(129,35),255,1)
    cv2.putText(ink,'20 - 0.1',(110,80),cv2.FONT_HERSHEY_SIMPLEX,.45,255,1)
    cv2.circle(ink,(140,108),7,255,1)
    cv2.line(ink,(134,114),(146,102),255,1)
    cv2.circle(ink,(174,100),2,255,1)
    cv2.line(ink,(190,105),(206,105),255,1)
    cv2.line(ink,(198,99),(198,111),255,1)
    cv2.line(ink,(190,118),(206,118),255,1)
    yy,xx=np.indices(ink.shape)
    for name,shadow,contrast in [('clean',0,160),('pale',0,55),('crease',65,160),('pale_crease',65,55)]:
        background=245-shadow*np.exp(-((xx-145-.3*yy)/18)**2)
        gray=np.clip(background-contrast*(ink>0),0,255).round().astype(np.uint8)
        yield name,gray,ink>0
