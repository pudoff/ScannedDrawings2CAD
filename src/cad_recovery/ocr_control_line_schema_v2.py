"""Explicit accepted, uncertain and absent slots for fixed control-title fields."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .family_catalog_v2 import canonical_sha, relative_path
from .label_schema_v1 import Pin
from .ocr_label_schema_v1 import Box, Size, _line, make_ocr_transform_v1
from .ocr_line_schema_v3 import StrictModel, VisualChecksV3, Text, Identifier, _identifier


FIELDS=('designation','part_name','adjacent_reference')


class ControlFieldV2(StrictModel):
    id: Identifier
    field: Literal['designation','part_name','adjacent_reference']
    parent_kind: Literal['left','right']
    source_box_xyxy: Box | None
    rotation_cw_degrees: Literal[0,90,180,270]
    raw_crop_pin: Pin | None
    image_pin: Pin | None
    decision: Literal['accepted_for_ocr_transcription','unresolved','absent']
    text: Annotated[str,Field(min_length=1,max_length=512)] | None
    reason: Text
    normalization: Text
    visual_checks: VisualChecksV3

    @field_validator('id')
    @classmethod
    def identifier(cls,value):return _identifier(value)

    @field_validator('text')
    @classmethod
    def single_line(cls,value):return None if value is None else _line(value)

    @field_validator('rotation_cw_degrees',mode='before')
    @classmethod
    def integer_rotation(cls,value):
        if type(value) is not int:raise ValueError('Quarter turn must be an explicit integer')
        return value

    @field_validator('reason','normalization')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Field explanation cannot be blank')
        return value

    @model_validator(mode='after')
    def decision_and_geometry(self):
        if self.decision=='absent':
            if any(v is not None for v in (self.text,self.source_box_xyxy,self.raw_crop_pin,self.image_pin)) or self.rotation_cw_degrees!=0:
                raise ValueError('Absent field cannot invent a crop, rotation or OCR target')
        else:
            if self.source_box_xyxy is None or self.raw_crop_pin is None or self.image_pin is None:
                raise ValueError('Present field requires exact geometry and both image pins')
            make_ocr_transform_v1(self.source_box_xyxy,self.rotation_cw_degrees)
        if self.decision=='accepted_for_ocr_transcription':
            if self.text is None or not all(self.visual_checks.model_dump().values()):
                raise ValueError('Accepted text requires a target and four positive visual checks')
        elif self.text is not None:raise ValueError('Uncertain or absent field must have text=null')
        return self


class ControlSourceV2(StrictModel):
    id: Identifier
    input_path: str
    input_sha256: str
    preparation: Pin
    source_size_px: Size
    fields: list[ControlFieldV2]=Field(min_length=3,max_length=3)
    scope_notes: list[Text]=Field(min_length=1,max_length=20)

    @field_validator('id')
    @classmethod
    def identifier(cls,value):return _identifier(value)

    @field_validator('input_path')
    @classmethod
    def relative(cls,value):return relative_path(value)

    @field_validator('input_sha256')
    @classmethod
    def revision(cls,value):return canonical_sha(value)

    @model_validator(mode='after')
    def complete_slots(self):
        if tuple(r.field for r in self.fields)!=FIELDS or len({r.id.casefold() for r in self.fields})!=3:
            raise ValueError('Every source requires exactly the three fixed field slots')
        w,h=self.source_size_px
        for row in self.fields:
            if row.source_box_xyxy is not None and (row.source_box_xyxy[2]>w or row.source_box_xyxy[3]>h):
                raise ValueError('Control field lies outside original source')
        return self


class ControlReviewV2(StrictModel):
    schema_version: Literal['ocr-control-line-reading/2.0']
    protocol: Pin
    crop_proposals: Pin
    reviewer: Literal['assistant']
    human_verified: Literal[False]
    ocr_run: Literal[False]
    native_cad_opened: Literal[False]
    training_started: Literal[False]
    reviewed_at_utc: str
    viewed_contacts: list[Pin]=Field(min_length=1,max_length=20)
    additional_viewed_images: list[Pin]=Field(max_length=20)
    sources: list[ControlSourceV2]=Field(min_length=1,max_length=23)

    @field_validator('reviewed_at_utc')
    @classmethod
    def timestamp(cls,value):
        if datetime.fromisoformat(value).tzinfo is None:raise ValueError('Line review needs an explicit timezone')
        return value

    @model_validator(mode='after')
    def unique_bindings(self):
        if len({r.id.casefold() for r in self.sources})!=len(self.sources) or len({r.input_path.casefold() for r in self.sources})!=len(self.sources):
            raise ValueError('Repeated source identity')
        identifiers=[r.id.casefold() for s in self.sources for r in s.fields]
        if len(set(identifiers))!=len(identifiers):raise ValueError('Field identity reused across sources')
        pins=[self.protocol,self.crop_proposals,*self.viewed_contacts,*self.additional_viewed_images]
        pins += [s.preparation for s in self.sources]
        pins += [p for s in self.sources for r in s.fields for p in (r.raw_crop_pin,r.image_pin) if p is not None]
        revisions={}
        for pin in pins:
            if pin.path in revisions and revisions[pin.path]!=pin.sha256:raise ValueError('Conflicting evidence revisions')
            revisions[pin.path]=pin.sha256
        return self
