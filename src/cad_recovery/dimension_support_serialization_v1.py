"""Lossless JSON scalar adapter for the preserved numerical evaluator."""
import numpy as np

def json_scalars(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_scalars(v) for k,v in value.items()}
    if isinstance(value,list):return [json_scalars(v) for v in value]
    return value
