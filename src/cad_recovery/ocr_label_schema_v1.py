"""A task-scoped OCR release, independent of physical drawing approval.

The schema validates claims and exact crop/quarter-turn coordinate arithmetic.
It does not authenticate files or independently re-review text: a consumer must
check the pinned review, source, image bytes and lineage before loading samples.
Frozen labels/1.0 observations remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .label_schema_v1 import Pin, read_json_v1


Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Pixel = Annotated[int, Field(strict=True, ge=0, le=100_000_000)]
Extent = Annotated[int, Field(strict=True, ge=1, le=100_000_000)]
Box = Annotated[list[Pixel], Field(min_length=4, max_length=4)]
Size = Annotated[list[Extent], Field(min_length=2, max_length=2)]
Coordinate = Annotated[float, Field(ge=-100_000_000, le=100_000_000, allow_inf_nan=False)]
Matrix = Annotated[list[Annotated[list[Coordinate], Field(min_length=3, max_length=3)]], Field(min_length=3, max_length=3)]


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True,
                              revalidate_instances='always', allow_inf_nan=False)

    @field_validator('*', mode='before')
    @classmethod
    def explicit_boolean_claims(cls, value, info):
        # Pydantic Literal[False/True] otherwise also accepts integer 0/1.
        names = {'train_only', 'training_eligible_for_task', 'training_started', 'augmentation_created',
                 'human_verified', 'pair_approved', 'cad_target_approved',
                 'physical_geometry_approved', 'physical_scale_confirmed'}
        if info.field_name in names and type(value) is not bool:
            raise ValueError('Approval and routing claims must use explicit JSON booleans')
        return value


def _box(value):
    if (not isinstance(value, list) or len(value) != 4 or
            any(type(v) is not int for v in value) or
            not 0 <= value[0] < value[2] <= 100_000_000 or
            not 0 <= value[1] < value[3] <= 100_000_000):
        raise ValueError('Expected an ordered half-open source-pixel box')
    return value


def _line(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 512 or
            value != value.strip() or unicodedata.normalize('NFC', value) != value or
            any(unicodedata.category(c).startswith('C') or c.isspace() and c != ' ' for c in value)):
        raise ValueError('OCR transcription must be a nonempty normalized single line without controls')
    return value


def make_ocr_transform_v1(source_box_xyxy, rotation_cw_degrees):
    """Exact mapping of original pixel centers to an unresampled image.

    Crop bounds are half-open. Clockwise 90 degrees corresponds to Pillow's
    ROTATE_270 transpose. No interpolation, scale fitting or arbitrary affine
    transform is permitted by this version.
    """
    x0, y0, x1, y1 = _box(source_box_xyxy)
    if type(rotation_cw_degrees) is not int or rotation_cw_degrees not in (0, 90, 180, 270):
        raise ValueError('Only exact clockwise quarter-turn rotations are supported')
    width, height = x1-x0, y1-y0
    if width*height > 4_194_304:
        raise ValueError('OCR crop exceeds the bounded image pixel budget')
    if rotation_cw_degrees == 0:
        forward = [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]
        inverse = [[1, 0, x0], [0, 1, y0], [0, 0, 1]]
    elif rotation_cw_degrees == 90:
        forward = [[0, -1, y1-1], [1, 0, -x0], [0, 0, 1]]
        inverse = [[0, 1, x0], [-1, 0, y1-1], [0, 0, 1]]
    elif rotation_cw_degrees == 180:
        forward = [[-1, 0, x1-1], [0, -1, y1-1], [0, 0, 1]]
        inverse = [[-1, 0, x1-1], [0, -1, y1-1], [0, 0, 1]]
    else:
        forward = [[0, 1, -y0], [-1, 0, x1-1], [0, 0, 1]]
        inverse = [[0, -1, x1-1], [1, 0, y0], [0, 0, 1]]
    return {'size_px': [height, width] if rotation_cw_degrees in (90, 270) else [width, height],
            'source_to_image': forward, 'image_to_source': inverse}


class PrintedComponentsV1(Model):
    """Parsed notation, never physical design dimensions or measured geometry."""
    nominal_value: Annotated[float, Field(gt=0, le=1e12)] | None = None
    chamfer_angle_deg: Annotated[float, Field(gt=0, lt=180)] | None = None
    chamfer_count: Annotated[int, Field(strict=True, gt=0, le=10000)] | None = None
    interpretation: Literal['printed_notation_not_physical_geometry'] = 'printed_notation_not_physical_geometry'


def parse_printed_components_v1(lines):
    """Parse this release's supported numeric/chamfer notation without units.

    This helper does not perform OCR and never silently normalizes x to ×.
    The visible cross glyph is normalized only by the explicit review policy.
    """
    if not isinstance(lines, list) or not lines:
        raise ValueError('Expected reviewed transcription lines')
    for line in lines:
        _line(line)
    number = r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?'
    if len(lines) == 1 and re.fullmatch(number, lines[0]):
        result = {'nominal_value': float(lines[0])}
    elif (len(lines) == 2 and
          (chamfer := re.fullmatch('('+number+')×('+number+')°', lines[0])) and
          (count := re.fullmatch(r'([1-9][0-9]*) фаски', lines[1]))):
        result = {'nominal_value': float(chamfer.group(1)), 'chamfer_angle_deg': float(chamfer.group(2)),
                  'chamfer_count': int(count.group(1))}
    else:
        raise ValueError('Unsupported reviewed printed notation for OCR release v1')
    return PrintedComponentsV1.model_validate(result).model_dump(mode='json')


class OcrAnnotationV1(Model):
    id: Identifier
    dimension_id: Identifier
    transcription_lines: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(min_length=1, max_length=2)
    sample_ids: list[Identifier] = Field(min_length=1, max_length=2)
    printed_components: PrintedComponentsV1
    normalization_policy: Literal['engineering_symbols_v1']
    normalization_notes: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def notation(self):
        if len(self.sample_ids) != len(self.transcription_lines) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('Every annotation line needs one distinct recognition sample')
        expected = parse_printed_components_v1(self.transcription_lines)
        if self.printed_components.model_dump(mode='json') != expected:
            raise ValueError('Printed components must follow the reviewed text, never a physical nominal')
        return self


class OcrSampleV1(Model):
    id: Identifier
    annotation_id: Identifier
    line_index: int = Field(strict=True, ge=0, le=1)
    text: str = Field(min_length=1, max_length=512)
    source_box_xyxy: Box
    parent_region_pin: Pin
    parent_region_box_xyxy: Box
    rotation_cw_degrees: Literal[0, 90, 180, 270]
    source_to_image: Matrix
    image_to_source: Matrix
    raw_crop_pin: Pin
    image_pin: Pin
    size_px: Size
    review_status: Literal['accepted_for_ocr_transcription'] = 'accepted_for_ocr_transcription'
    reviewer: Literal['assistant'] = 'assistant'
    human_verified: Literal[False] = False
    physical_geometry_approved: Literal[False] = False

    @field_validator('text')
    @classmethod
    def single_line(cls, value):
        return _line(value)

    @field_validator('rotation_cw_degrees', mode='before')
    @classmethod
    def integral_rotation(cls, value):
        if type(value) is not int:
            raise ValueError('Rotation must be an explicit integer quarter turn')
        return value

    @model_validator(mode='after')
    def exact_crop_transform(self):
        source, parent = _box(self.source_box_xyxy), _box(self.parent_region_box_xyxy)
        if not (parent[0] <= source[0] < source[2] <= parent[2] and
                parent[1] <= source[1] < source[3] <= parent[3]):
            raise ValueError('Line crop must lie inside its reviewed parent region')
        expected = make_ocr_transform_v1(source, self.rotation_cw_degrees)
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError('Image transform must exactly match pixel-center crop and quarter-turn rotation')
        return self


class OcrLineageV1(Model):
    group_id: Identifier
    family_ids: list[Identifier] = Field(max_length=1000)
    train_only: bool
    historical_reserved_split: Literal['train'] | None
    split: Literal['train'] = 'train'
    catalog: Pin
    split_policy: Pin

    @model_validator(mode='after')
    def unique_families(self):
        if len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError('Lineage family IDs must be distinct')
        return self


class UnknownPhysicalFieldsV1(Model):
    nominal_mm: None = None
    scale_mm_per_pixel: None = None
    geometry: None = None
    supports: None = None


class OcrLabelPackageV1(Model):
    schema_version: Literal['ocr-labels/1.0'] = 'ocr-labels/1.0'
    purpose: Literal['ocr_transcription_for_development_train'] = 'ocr_transcription_for_development_train'
    id: Identifier
    source: Pin
    source_size_px: Size
    review: Pin
    dependencies: list[Pin] = Field(min_length=1, max_length=200)
    lineage: OcrLineageV1
    annotations: list[OcrAnnotationV1] = Field(min_length=1, max_length=4096)
    samples: list[OcrSampleV1] = Field(min_length=1, max_length=8192)
    task: Literal['ocr_recognition'] = 'ocr_recognition'
    release_scope: Literal['reviewed_transcriptions_only'] = 'reviewed_transcriptions_only'
    training_eligible_for_task: Literal[True] = True
    training_started: Literal[False] = False
    augmentation_created: Literal[False] = False
    human_verified: Literal[False] = False
    pair_approved: Literal[False] = False
    cad_target_approved: Literal[False] = False
    physical_geometry_approved: Literal[False] = False
    physical_scale_confirmed: Literal[False] = False
    physical_fields: UnknownPhysicalFieldsV1 = Field(default_factory=UnknownPhysicalFieldsV1)
    limitations: list[Annotated[str, Field(min_length=1, max_length=4000)]] = Field(min_length=1, max_length=50)

    @model_validator(mode='after')
    def package_links(self):
        annotations = {a.id: a for a in self.annotations}
        samples = {s.id: s for s in self.samples}
        if len(annotations) != len(self.annotations) or len(samples) != len(self.samples):
            raise ValueError('Annotation and sample IDs must be unique')
        if len({a.dimension_id for a in self.annotations}) != len(self.annotations):
            raise ValueError('A source dimension may have only one annotation record')
        links = [sid for annotation in self.annotations for sid in annotation.sample_ids]
        if len(links) != len(set(links)) or set(links) != set(samples):
            raise ValueError('Each recognition sample belongs to exactly one reviewed annotation line')
        dependency_map = {p.path: p.sha256 for p in self.dependencies}
        if len(dependency_map) != len(self.dependencies):
            raise ValueError('Dependency paths must be unique')
        for pin in (self.review, self.lineage.catalog, self.lineage.split_policy):
            if dependency_map.get(pin.path) != pin.sha256:
                raise ValueError('Review and lineage references must occur in pinned dependencies')
        all_pins = [self.source, *self.dependencies]
        crops = set()
        for sample in self.samples:
            if sample.annotation_id not in annotations:
                raise ValueError('Sample refers to a missing annotation')
            annotation = annotations[sample.annotation_id]
            if (sample.line_index >= len(annotation.sample_ids) or annotation.sample_ids[sample.line_index] != sample.id or
                    annotation.transcription_lines[sample.line_index] != sample.text):
                raise ValueError('Sample text and line order must equal the reviewed annotation')
            if dependency_map.get(sample.parent_region_pin.path) != sample.parent_region_pin.sha256:
                raise ValueError('Parent raster region must be pinned in dependencies')
            if sample.parent_region_box_xyxy[2] > self.source_size_px[0] or sample.parent_region_box_xyxy[3] > self.source_size_px[1]:
                raise ValueError('Reviewed region lies outside the original source image')
            crop = tuple(sample.source_box_xyxy)
            if crop in crops:
                raise ValueError('Repeated source crop, even rotated, cannot count as another OCR observation')
            crops.add(crop)
            all_pins.extend((sample.parent_region_pin, sample.raw_crop_pin, sample.image_pin))
        actual = {}
        for pin in all_pins:
            if pin.path in actual and actual[pin.path] != pin.sha256:
                raise ValueError('One artifact path cannot name different content hashes')
            actual[pin.path] = pin.sha256
        return self


def read_ocr_labels_v1(value):
    """Validate a fresh package copy; actual-byte/source/lineage checks are external."""
    if isinstance(value, (str, Path)):
        value = read_json_v1(value, max_bytes=8*1024**2)
    if isinstance(value, OcrLabelPackageV1):
        value = value.model_dump(mode='json')
    return OcrLabelPackageV1.model_validate(value)
