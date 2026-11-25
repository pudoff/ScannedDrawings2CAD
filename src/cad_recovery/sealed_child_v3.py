"""Retain the V2 seal/coverage contract; use the corrected phase lifecycle."""
from .sealed_child_v2 import finalize
from .child_lifecycle_v1 import child_main as correlated_child


def child_main(path,operation,*,seal=None):
    def execute(request,progress):
        operation(request,progress);finalize(request,progress,seal=seal)
    return correlated_child(path,execute)
