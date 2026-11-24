"""Strict source-only OCR line reviews, without family or split approval.

This schema checks authored decisions, bounded crop geometry and claim scope.
It does not authenticate actual bytes, verify visual truth, or release labels;
those checks belong to the pinned release builder and consumer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .family_catalog_v2 import canonical_sha, relative_path
from .label_schema_v1 import Pin
from .ocr_label_schema_v1 import Box, Size, _line, make_ocr_transform_v1


Text = Annotated[str, Field(min_length=1, max_length=4096)]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]
FALSE_CLAIMS = frozenset({'human_verified', 'ocr_run', 'native_cad_opened', 'training_started'})


def _identifier(value):
    # The IDs become PNG basenames; also avoid Windows device filenames.
    if (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is None or '..' in value
            or value.endswith('.') or value.split('.')[0].upper() in
            {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('Review identities must be portable artifact basenames')
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True,
                              revalidate_instances='always', allow_inf_nan=False)

    @field_validator('*', mode='before')
    @classmethod
    def explicit_claims(cls, value, info):
        if info.field_name in FALSE_CLAIMS and type(value) is not bool:
            raise ValueError('Review claims require explicit JSON booleans')
        return value


class VisualChecksV3(StrictModel):
    full_text_visible: bool
    text_matches: bool
    no_line_carrier: bool
    exact_rotation: bool


class LineV3(StrictModel):
    id: Identifier
    annotation_id: Identifier
    role: Text
    parent_kind: Literal['title', 'context']
    source_box_xyxy: Box
    rotation_cw_degrees: Literal[0, 90, 180, 270]
    text: Annotated[str, Field(min_length=1, max_length=512)] | None
    decision: Literal['accepted_for_ocr_transcription', 'unresolved']
    reason: Text
    normalization: Text
    raw_crop_pin: Pin | None
    image_pin: Pin | None
    visual_checks: VisualChecksV3

    @field_validator('id', 'annotation_id')
    @classmethod
    def portable_id(cls, value):
        return _identifier(value)

    @field_validator('role', 'reason', 'normalization')
    @classmethod
    def nonempty_notes(cls, value):
        if not value.strip():
            raise ValueError('Review role, reason and normalization cannot be blank')
        return value

    @field_validator('text')
    @classmethod
    def normalized_single_line(cls, value):
        return None if value is None else _line(value)

    @field_validator('rotation_cw_degrees', mode='before')
    @classmethod
    def integral_rotation(cls, value):
        if type(value) is not int:
            raise ValueError('Rotation requires an explicit integer quarter turn')
        return value

    @model_validator(mode='after')
    def crop_and_decision(self):
        make_ocr_transform_v1(self.source_box_xyxy, self.rotation_cw_degrees)
        if self.decision == 'accepted_for_ocr_transcription':
            if self.text is None or self.raw_crop_pin is None or self.image_pin is None:
                raise ValueError('Accepted text requires a transcription and both exact image pins')
            if not all(self.visual_checks.model_dump().values()):
                raise ValueError('Accepted text requires all four positive visual checks')
        elif self.text is not None:
            raise ValueError('Unresolved observations must not contain an OCR target')
        return self


class SourceV3(StrictModel):
    id: Identifier
    input_path: Annotated[str, Field(min_length=1, max_length=4096)]
    input_sha256: Annotated[str, Field(min_length=64, max_length=64)]
    preparation: Pin
    source_size_px: Size
    lines: list[LineV3] = Field(min_length=1, max_length=64)
    scope_notes: list[Text] = Field(min_length=1, max_length=50)

    @field_validator('id')
    @classmethod
    def portable_id(cls, value):
        return _identifier(value)

    @field_validator('input_path')
    @classmethod
    def dataset_relative(cls, value):
        return relative_path(value)

    @field_validator('input_sha256')
    @classmethod
    def source_revision(cls, value):
        return canonical_sha(value)

    @field_validator('scope_notes')
    @classmethod
    def nonempty_scope_notes(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError('Scope notes cannot be blank')
        return values

    @model_validator(mode='after')
    def source_geometry(self):
        if len({line.id for line in self.lines}) != len(self.lines):
            raise ValueError('Line identities must be distinct within a source')
        width, height = self.source_size_px
        boxes, annotations = set(), {}
        for line in self.lines:
            box = line.source_box_xyxy
            if box[2] > width or box[3] > height:
                raise ValueError('Line crop lies outside its original source extent')
            if line.annotation_id in annotations and annotations[line.annotation_id] != line.parent_kind:
                raise ValueError('An annotation belongs to one reviewed parent region')
            annotations[line.annotation_id] = line.parent_kind
            if line.decision == 'accepted_for_ocr_transcription':
                key = (tuple(box), line.rotation_cw_degrees)
                if key in boxes:
                    raise ValueError('Duplicate accepted crop and rotation')
                boxes.add(key)
        return self


class ReviewV3(StrictModel):
    schema_version: Literal['ocr-line-reading/3.0']
    protocol: Pin
    reviewer: Literal['assistant']
    human_verified: Literal[False]
    ocr_run: Literal[False]
    native_cad_opened: Literal[False]
    training_started: Literal[False]
    reviewed_at_utc: Annotated[str, Field(min_length=1, max_length=64)]
    sources: list[SourceV3] = Field(min_length=1, max_length=6)

    @field_validator('reviewed_at_utc')
    @classmethod
    def timezone_timestamp(cls, value):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError('Review timestamp must be ISO-8601 with a timezone') from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('Review timestamp must include a timezone')
        return value

    @model_validator(mode='after')
    def distinct_identities(self):
        if (len({source.id for source in self.sources}) != len(self.sources)
                or len({source.input_path for source in self.sources}) != len(self.sources)):
            raise ValueError('Review sources need unique identities and paths')
        line_ids, annotations, revisions = set(), {}, {}
        pins = [self.protocol]
        for source in self.sources:
            pins.append(source.preparation)
            for line in source.lines:
                if line.id in line_ids:
                    raise ValueError('Line identity is reused across sources')
                line_ids.add(line.id)
                if line.annotation_id in annotations and annotations[line.annotation_id] != source.id:
                    raise ValueError('Annotation identity is reused across sources')
                annotations[line.annotation_id] = source.id
                pins.extend(pin for pin in (line.raw_crop_pin, line.image_pin) if pin is not None)
        for pin in pins:
            if pin.path in revisions and revisions[pin.path] != pin.sha256:
                raise ValueError('Review refers to conflicting revisions of one artifact')
            revisions[pin.path] = pin.sha256
        return self
