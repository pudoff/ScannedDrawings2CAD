"""Versioned field-level labels. V1 is an observation corpus, never a release."""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Status = Literal['verified', 'observation', 'unknown', 'rejected']
Domain = Literal['customer_observation', 'analytical_control']
Kind = Literal['view', 'title_block', 'geometry', 'arrow', 'text', 'dimension', 'anchor', 'contour', 'hole', 'hatch']
SHA = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
FIELDS = {
    'view': {'extent', 'printed_label', 'scale_caption', 'physical_scale'},
    'title_block': {'readings', 'native_text', 'identity'},
    'geometry': {'shape', 'role', 'raster_projection', 'physical_geometry', 'binding'},
    'arrow': {'contact_interval', 'glyph', 'measured_support'},
    'text': {'transcription', 'box', 'native_text'},
    'dimension': {'notation', 'printed_nominal', 'physical_nominal', 'tolerances', 'supports', 'view_scale', 'numeric_consistency'},
    'anchor': {'native_point', 'raster_interval', 'raster_projection', 'physical_point', 'binding'},
    'contour': {'edges', 'closure', 'role', 'physical_boundary'},
    'hole': {'boundary', 'parent', 'physical_boundary'},
    'hatch': {'boundary', 'pattern', 'region'},
}


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True, revalidate_instances='always', allow_inf_nan=False)


class Pin(Model):
    path: str = Field(min_length=1)
    sha256: SHA

    @field_validator('path')
    @classmethod
    def relative(cls, value):
        p = PurePosixPath(value)
        if '\\' in value or ':' in value or p.is_absolute() or '..' in p.parts or p.as_posix() != value:
            raise ValueError('Artifact paths must be normalized workspace-relative paths')
        return value


class Evidence(Model):
    artifact: Pin
    locator: str = Field(min_length=1)
    domain: Domain
    method: Literal['raster_transcription', 'author_interval', 'native_observation', 'derived_projection',
                    'review_record', 'analytical_exact', 'unreviewed']
    statement: str = Field(min_length=1)


class Quality(Model):
    basis: str = Field(min_length=1)
    accuracy_bound: float | None = Field(default=None, ge=0, le=1e12)
    bound_unit: Literal['source_pixel', 'view_pixel', 'native_unit', 'paper_mm', 'model_mm', 'degree'] | None = None
    statistical_confidence: None = None

    @model_validator(mode='after')
    def bound_pair(self):
        if (self.accuracy_bound is None) != (self.bound_unit is None):
            raise ValueError('An accuracy bound needs its unit')
        return self


def finite_json(value):
    """Reject nonfinite nested payloads, including overflow decoded as Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite label value')
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)
    return value


class LabelField(Model):
    status: Status
    value: JsonValue = None
    unit: Literal['text', 'source_pixel', 'view_pixel', 'native_unit', 'paper_mm', 'model_mm',
                  'degree', 'printed_value', 'dimensionless', 'structured'] = 'structured'
    frame_id: str | None = None
    evidence: list[Evidence] = Field(min_length=1)
    quality: Quality

    @field_validator('value')
    @classmethod
    def finite(cls, value):
        return finite_json(value)

    @model_validator(mode='after')
    def claims(self):
        if (self.status == 'unknown') != (self.value is None):
            raise ValueError('Unknown fields have null values; known/rejected fields retain a value')
        if self.status == 'verified' and any(e.domain != 'analytical_control' or e.method != 'analytical_exact' for e in self.evidence):
            raise ValueError('V1 verified fields require analytical construction evidence')
        if self.quality.accuracy_bound is not None and self.status != 'verified':
            raise ValueError('Authored intervals and comparison tolerances are not accuracy certificates')
        return self


class LabelObject(Model):
    id: str = Field(min_length=1)
    kind: Kind
    view_id: str | None = None
    fields: dict[str, LabelField]

    @model_validator(mode='after')
    def expected_fields(self):
        if set(self.fields) != FIELDS[self.kind]:
            raise ValueError('Every object must state the complete field set for its kind')
        return self


class Frame(Model):
    id: str = Field(min_length=1)
    unit: Literal['source_pixel', 'view_pixel', 'native_unit', 'paper_mm', 'model_mm', 'analytic_pixel']
    axes: Literal['x_right_y_down', 'x_right_y_up']
    source_sha256: SHA
    physical_scale_confirmed: bool = False
    meaning: str = Field(min_length=1)


class Transform(Model):
    id: str
    from_frame: str
    to_frame: str
    matrix: list[list[float]]
    inverse: list[list[float]]
    status: Literal['observation', 'verified']
    evidence: Evidence

    @model_validator(mode='after')
    def affine(self):
        a, b = np.asarray(self.matrix), np.asarray(self.inverse)
        if (a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all()
                or max(np.abs(a).max(), np.abs(b).max()) > 1e12
                or not np.array_equal(a[2], [0, 0, 1]) or not np.array_equal(b[2], [0, 0, 1])
                or not np.allclose(a @ b, np.eye(3), rtol=0, atol=1e-8)
                or not np.allclose(b @ a, np.eye(3), rtol=0, atol=1e-8)):
            raise ValueError('Transforms require bounded invertible affine matrices with matching inverses')
        if self.from_frame == self.to_frame:
            raise ValueError('A transform must connect distinct frames')
        if self.status == 'verified' and (self.evidence.domain != 'analytical_control' or self.evidence.method != 'analytical_exact'):
            raise ValueError('Verified transforms require analytical evidence')
        return self


class Coverage(Model):
    kind: Kind
    state: Literal['partial_observations', 'analytical_control', 'unreviewed']
    object_count: int = Field(ge=0)
    reason: str = Field(min_length=1)


class Lineage(Model):
    group_id: str | None = None
    family_id: str | None = None
    train_only: bool = False
    historical_reserved_split: Literal['train', 'validation', 'test'] | None = None
    catalog: Pin | None = None
    split: None = None
    augmentation_state: Literal['not_created_before_split'] = 'not_created_before_split'


class LabelPackage(Model):
    schema_version: Literal['labels/1.0'] = 'labels/1.0'
    purpose: Literal['field_observations_and_analytical_controls'] = 'field_observations_and_analytical_controls'
    id: str
    domain: Domain
    source: Pin
    native_target: Pin | None = None
    lineage: Lineage
    frames: list[Frame] = Field(min_length=1, max_length=1000)
    transforms: list[Transform] = Field(max_length=2000)
    objects: list[LabelObject] = Field(max_length=20000)
    coverage: list[Coverage]
    dependencies: list[Pin] = Field(min_length=1, max_length=100)
    assembly_inputs: dict[str, Pin]
    limitations: list[str] = Field(min_length=1)
    annotation_approved: Literal[False] = False
    training_eligible: Literal[False] = False
    split: None = None

    @model_validator(mode='after')
    def consistency(self):
        frames = {f.id: f for f in self.frames}
        objects = {o.id: o for o in self.objects}
        if len(frames) != len(self.frames) or len(objects) != len(self.objects):
            raise ValueError('Frame/object IDs must be unique')
        if len({t.id for t in self.transforms}) != len(self.transforms):
            raise ValueError('Transform IDs must be unique')
        if any(f.source_sha256 != self.source.sha256 for f in self.frames):
            raise ValueError('Every coordinate frame must identify the package source')
        customer = self.domain == 'customer_observation'
        if customer and (self.native_target is None or not self.lineage.group_id or self.lineage.catalog is None):
            raise ValueError('Customer observations require exact native target and catalog lineage')
        if not customer and (self.native_target is not None or self.lineage.group_id is not None or self.lineage.catalog is not None):
            raise ValueError('Analytical controls cannot claim customer lineage')
        if customer and any(f.unit in ('model_mm', 'analytic_pixel') or f.physical_scale_confirmed for f in self.frames):
            raise ValueError('Customer physical coordinate labels are pending in v1')
        pins = {(p.path, p.sha256) for p in [self.source, *self.dependencies, *([self.native_target] if self.native_target else [])]}
        if {(p.path, p.sha256) for p in self.assembly_inputs.values()} != {(p.path, p.sha256) for p in self.dependencies}:
            raise ValueError('Assembly inputs must exactly match the pinned dependencies')
        if not customer and self.source != self.assembly_inputs.get('control_spec'):
            raise ValueError('Analytical source must be the pinned construction spec')
        views = {o.id for o in self.objects if o.kind == 'view'}
        for obj in self.objects:
            if obj.view_id is not None and obj.view_id not in views:
                raise ValueError('Object refers to a missing view')
            for name, field in obj.fields.items():
                if field.frame_id is not None and field.frame_id not in frames:
                    raise ValueError('Field refers to a missing coordinate frame')
                if field.frame_id is not None and field.unit in ('source_pixel', 'view_pixel', 'native_unit', 'paper_mm', 'model_mm'):
                    expected = frames[field.frame_id].unit
                    if field.unit != expected and not (field.unit == 'source_pixel' and expected == 'analytic_pixel'):
                        raise ValueError('Field coordinate unit differs from its frame')
                if customer and (field.status == 'verified' or (field.unit == 'model_mm' and field.status != 'unknown')):
                    raise ValueError('No customer ground truth may be minted by this observation schema')
                if customer and name in ('physical_scale', 'physical_geometry', 'physical_nominal', 'physical_point', 'physical_boundary') and field.status != 'unknown':
                    raise ValueError('Physical customer fields must remain unknown')
                for evidence in field.evidence:
                    if evidence.domain != self.domain or (evidence.artifact.path, evidence.artifact.sha256) not in pins:
                        raise ValueError('Field evidence must be pinned and scoped to this package')
        for t in self.transforms:
            if t.from_frame not in frames or t.to_frame not in frames:
                raise ValueError('Transform refers to a missing frame')
            if (t.evidence.domain != self.domain or (t.evidence.artifact.path, t.evidence.artifact.sha256) not in pins
                    or customer and t.status != 'observation'):
                raise ValueError('Transform evidence scope/pin does not match package')
        coverage = {c.kind: c for c in self.coverage}
        if len(coverage) != len(self.coverage) or set(coverage) != set(FIELDS):
            raise ValueError('Coverage must enumerate every label kind, including missing ones')
        for kind, item in coverage.items():
            count = sum(o.kind == kind for o in self.objects)
            if count != item.object_count or (count == 0) != (item.state == 'unreviewed'):
                raise ValueError('Coverage counts must agree with objects')
            if customer and item.state == 'analytical_control':
                raise ValueError('Control coverage is not customer approval')
        return self


def read_json_v1(path, *, max_bytes=64 * 1024**2):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('JSON input exceeds the size budget')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    return finite_json(value)


def read_labels_v1(value: LabelPackage | dict | str | Path) -> LabelPackage:
    if isinstance(value, (str, Path)):
        value = read_json_v1(value)
    if isinstance(value, LabelPackage):
        value = value.model_dump(mode='json')
    return LabelPackage.model_validate(value)


def transform_points_v1(package, transform_id, points, *, inverse=False):
    """Coordinate arithmetic only: the returned status never upgrades its evidence."""
    package = read_labels_v1(package)
    found = [t for t in package.transforms if t.id == transform_id]
    if len(found) != 1:
        raise ValueError('Unknown transform')
    t = found[0]
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or (p.size and np.abs(p).max() > 1e12):
        raise ValueError('Expected bounded finite 2D points')
    mapped = np.column_stack((p, np.ones(len(p)))) @ np.asarray(t.inverse if inverse else t.matrix).T
    if not np.isfinite(mapped).all() or (mapped.size and np.abs(mapped).max() > 1e12):
        raise ValueError('Mapped coordinates exceed the representation')
    return {'points': mapped[:, :2].tolist(), 'frame_id': t.from_frame if inverse else t.to_frame,
            'status': t.status, 'evidence': t.evidence.model_dump(mode='json'), 'training_eligible': False}
