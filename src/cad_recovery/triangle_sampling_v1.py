"""Bounded, deterministic area sampling without a triangle-sized bitmap.

The returned fraction estimates area coverage. It does not prove that every
pixel is black, and is never a calibrated probability of a correct arrow.
"""
import math
import numpy as np
from . import dimension_association_v3 as v3
from .dimension_support_v1 import POLICY as PIXEL_POLICY

POLICY = {'version':'triangle-sampling/1.0','grid_side':32,'maximum_samples':1024,
          'maximum_coordinate_px':1_000_000,'minimum_arrow_ink_fraction':.85,
          'area_fraction_is_estimate':True,'production_admitted':False}


def points(polygon):
    polygon = v3._polygon(polygon, 'Triangle sampling polygon')
    if polygon.shape != (3, 2):
        raise ValueError('A triangle is required')
    if np.abs(polygon).max() > POLICY['maximum_coordinate_px']:
        raise v3._SearchLimit()
    side = POLICY['grid_side']
    if side < 1 or side * side > POLICY['maximum_samples']:
        raise v3._SearchLimit()
    # Square-to-triangle mapping has uniform area density. Midpoints avoid
    # intentionally over-weighting the boundary on small raster triangles.
    u, v = np.meshgrid((np.arange(side) + .5) / side, (np.arange(side) + .5) / side)
    r = np.sqrt(u.ravel());v = v.ravel()
    return ((1-r[:,None])*polygon[0] + (r*(1-v))[:,None]*polygon[1]
            + (r*v)[:,None]*polygon[2])


def fraction(pixels, arrow):
    samples = points(arrow['polygon_px'])
    if pixels.samples + len(samples) > PIXEL_POLICY['maximum_pixel_samples']:
        raise v3._SearchLimit()
    pixels.samples += len(samples)
    q = np.rint(samples).astype(int);height,width = pixels.distance.shape
    inside = (q >= 0).all(axis=1) & (q[:,0] < width) & (q[:,1] < height)
    ink = np.zeros(len(q),bool)
    ink[inside] = pixels.distance[q[inside,1],q[inside,0]] <= .5
    return float(ink.mean())
