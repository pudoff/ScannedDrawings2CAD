"""Reuse exact signed distances when clipping bounded native-frame polygons."""
import numpy as np


def clip_polygon(polygon, normal, bound):
    # Keep each scalar dot/subtraction identical to the frozen implementation.
    # Vectorized dot products can change rounding at a quantization boundary.
    polygon = np.asarray(polygon)
    distances = [float(point @ normal-bound) for point in polygon]
    inside = [distance <= 0 for distance in distances]
    if all(inside):
        return np.asarray(polygon, float).reshape(-1, 2).copy()
    if not any(inside):
        return np.empty((0, 2), float)
    output = []
    for index, (a, da) in enumerate(zip(polygon, distances)):
        following = (index+1) % len(polygon)
        if inside[index]:
            output.append(a)
        if inside[index] != inside[following]:
            b = polygon[following]; db = distances[following]
            output.append(a+(b-a)*(da/(da-db)))
    return np.asarray(output, float).reshape(-1, 2)
