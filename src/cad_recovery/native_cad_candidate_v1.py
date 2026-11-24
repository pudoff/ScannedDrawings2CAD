"""Candidate geometry with unchanged OCR planning, in a dedicated child process.

The geometry implementation temporarily patches legacy modules. Call this only
in a fresh, single-job process, never in the shared service worker.
"""
from .native_annotation_export_v2 import export_annotations
from .cad_axis_candidate_export_v1 import export_objects


def export_candidate(packet, annotations, target, *, protected=None, render=True):
    return export_annotations(packet, annotations, target, protected=protected,
                              render=render, geometry_export=export_objects)
