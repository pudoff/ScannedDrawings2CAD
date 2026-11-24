"""Public registration guards over the frozen native-table experiment.

Old PDF rendered pixels are not native scan pixels. Such conversions need an
explicit registration and must not silently reuse the raster identity matrix.
"""
from pathlib import Path
from .dataset_layout_v1 import read
from . import native_table_pipeline_v1 as previous


def prepare_review(source, baseline, previous_review, output):
    prior = read(previous_review)
    if (Path(source).suffix.lower() == '.pdf'
            or prior.get('source_coordinate_space') != 'original_raster_pixels'):
        raise ValueError('Legacy review registration requires original raster pixels; PDF rendered/native registration is not qualified')
    return previous.prepare_review(source, baseline, previous_review, output)


def run(source, baseline, review_path, source_manifest, output, *, engine):
    if Path(source).suffix.lower() == '.pdf':
        report = read(Path(baseline)/'report.json')
        if report.get('schema_version') != 'native-cad-pipeline/2.0':
            raise ValueError('Native PDF tables require a native CAD baseline and explicit native-page review')
    return previous.run(source, baseline, review_path, source_manifest, output, engine=engine)
