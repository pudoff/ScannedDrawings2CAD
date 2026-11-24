"""Multi-source reviewed OCR labels with exact printed notation, not geometry.

Actual-byte authentication, source-pixel replay, review decisions and the full
lineage allocation are enforced by the release consumer. This schema checks
their explicit scope and links without manufacturing a physical measurement.
Frozen ocr-labels/1.0 is reused for exact crop/quarter-turn sample validation.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .label_schema_v1 import Pin, read_json_v1
from .ocr_label_schema_v1 import (Identifier, Model, OcrLineageV1, OcrSampleV1,
                                   Size, UnknownPhysicalFieldsV1, _line,
                                   make_ocr_transform_v1)


NUMBER = r'(?:0|[1-9][0-9]*)(?:[,.][0-9]+)?'


class PrintedNumberV2(Model):
    raw: Annotated[str, Field(min_length=1, max_length=64)]
    decimal: Annotated[str, Field(min_length=1, max_length=64)]

    @model_validator(mode='after')
    def exact_decimal(self):
        if (re.fullmatch(NUMBER, self.raw) is None or
                self.decimal != self.raw.replace(',', '.') or
                Decimal(self.decimal) > Decimal('1e12')):
            raise ValueError('Printed number requires its exact unchanged token and decimal representation')
        return self


class PrintedComponentsV2(Model):
    status: Literal['parsed', 'unresolved']
    notation_kind: Literal['number', 'reference_number', 'chamfer', 'radius', 'diameter', 'thread', 'unresolved']
    nominal: PrintedNumberV2 | None = None
    chamfer_angle: PrintedNumberV2 | None = None
    quantity: Annotated[int, Field(strict=True, ge=0, le=10000)] | None = None
    quantity_suffix: Literal['фаски', 'отв', 'отв.', 'рад.'] | None = None
    fit_designator: Annotated[str, Field(pattern=r'^[A-Za-z][0-9]{1,2}$')] | None = None
    thread_designator: Annotated[str, Field(pattern=r'^M(?:0|[1-9][0-9]*)(?:[,.][0-9]+)?–[1-9][0-9]?[A-Z]$')] | None = None
    reference_marker: Literal['*'] | None = None
    unresolved_reason: Annotated[str, Field(min_length=1, max_length=4000)] | None = None
    physical_unit: None = None
    numeric_tolerance_limits: None = None
    thread_pitch: None = None
    interpretation: Literal['printed_notation_not_physical_geometry'] = 'printed_notation_not_physical_geometry'

    @model_validator(mode='after')
    def unresolved_is_unknown(self):
        if self.status == 'unresolved':
            if (self.notation_kind != 'unresolved' or not self.unresolved_reason or any(
                    getattr(self, key) is not None for key in ('nominal', 'chamfer_angle', 'quantity', 'quantity_suffix',
                                                             'fit_designator', 'thread_designator', 'reference_marker'))):
                raise ValueError('Unresolved notation must not supply guessed printed components')
        elif self.notation_kind == 'unresolved' or self.unresolved_reason is not None or self.nominal is None:
            raise ValueError('Parsed notation requires a printed nominal and no unresolved claim')
        if (self.quantity is None) != (self.quantity_suffix is None):
            raise ValueError('Printed quantity and its literal suffix belong together')
        return self


def _number(raw):
    return PrintedNumberV2(raw=raw, decimal=raw.replace(',', '.')).model_dump(mode='json')


def parse_printed_components_v2(lines):
    """Parse explicit canonical tokens; never infer a unit, fit limit or pitch.

    Decimal commas and typography remain in the transcription. Numbers also
    retain exact decimal strings, avoiding binary-float loss. Unsupported text
    remains an eligible *text* observation only if independently reviewed; its
    notation fields are explicitly unresolved. No x/х, Ø/Ф or hyphen aliases
    are silently changed into ×, ⌀ or an en dash.
    """
    if not isinstance(lines, list) or not 1 <= len(lines) <= 2:
        raise ValueError('Expected one or two reviewed OCR lines')
    for line in lines:
        _line(line)
    first = lines[0]
    data = None
    if match := re.fullmatch('('+NUMBER+r')(\*)?', first):
        data = {'notation_kind': 'reference_number' if match[2] else 'number',
                'nominal': _number(match[1]), 'reference_marker': match[2]}
    elif match := re.fullmatch('('+NUMBER+')×('+NUMBER+')°', first):
        data = {'notation_kind': 'chamfer', 'nominal': _number(match[1]), 'chamfer_angle': _number(match[2])}
    elif match := re.fullmatch('R('+NUMBER+')', first):
        data = {'notation_kind': 'radius', 'nominal': _number(match[1])}
    elif match := re.fullmatch('⌀('+NUMBER+r')([A-Za-z][0-9]{1,2})?', first):
        data = {'notation_kind': 'diameter', 'nominal': _number(match[1]), 'fit_designator': match[2]}
    elif match := re.fullmatch('M('+NUMBER+r')–([1-9][0-9]?[A-Z])', first):
        data = {'notation_kind': 'thread', 'nominal': _number(match[1]), 'thread_designator': first}
    if data is not None and len(lines) == 2:
        match = re.fullmatch(r'(0|[1-9][0-9]*) (фаски|отв\.?|рад\.)', lines[1])
        expected_suffixes = {'chamfer': {'фаски'}, 'radius': {'рад.'},
                             'diameter': {'отв', 'отв.'}, 'thread': {'отв', 'отв.'}}
        if match and match[2] in expected_suffixes.get(data['notation_kind'], set()):
            data.update(quantity=int(match[1]), quantity_suffix=match[2])
        else:
            data = None
    if data is None:
        return PrintedComponentsV2(status='unresolved', notation_kind='unresolved',
            unresolved_reason='Unsupported or ambiguous printed notation; reviewed text is retained without inferred components.').model_dump(mode='json')
    return PrintedComponentsV2(status='parsed', **data).model_dump(mode='json')


class OcrSourceV2(Model):
    id: Identifier
    source: Pin
    source_size_px: Size
    preparation: Pin
    review: Pin
    lineage: OcrLineageV1


class OcrAnnotationV2(Model):
    id: Identifier
    source_id: Identifier
    dimension_id: Identifier
    transcription_lines: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(min_length=1, max_length=2)
    sample_ids: list[Identifier] = Field(min_length=1, max_length=2)
    printed_components: PrintedComponentsV2
    normalization_policy: Literal['engineering_symbols_v2'] = 'engineering_symbols_v2'
    normalization_notes: Annotated[str, Field(min_length=1, max_length=4000)]
    review: Pin

    @model_validator(mode='after')
    def notation(self):
        if len(self.sample_ids) != len(self.transcription_lines) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('Each annotation line requires one distinct sample')
        if self.printed_components.model_dump(mode='json') != parse_printed_components_v2(self.transcription_lines):
            raise ValueError('Printed components differ from the explicitly reviewed text')
        return self


class OcrSampleV2(OcrSampleV1):
    source_id: Identifier


class UnresolvedOcrAnnotationV2(Model):
    id: Identifier
    source_id: Identifier
    dimension_id: Identifier
    source_reading: Annotated[str, Field(min_length=1, max_length=4000)]
    reason: Annotated[str, Field(min_length=1, max_length=4000)]
    review: Pin
    decision: Literal['unresolved'] = 'unresolved'


class OcrLabelPackageV2(Model):
    schema_version: Literal['ocr-labels/2.0'] = 'ocr-labels/2.0'
    purpose: Literal['ocr_transcription_for_development_train'] = 'ocr_transcription_for_development_train'
    id: Identifier
    parent_release: Pin
    inherited_sample_ids: list[Identifier] = Field(min_length=1, max_length=8192)
    source_records: list[OcrSourceV2] = Field(min_length=1, max_length=64)
    dependencies: list[Pin] = Field(min_length=1, max_length=1000)
    annotations: list[OcrAnnotationV2] = Field(min_length=1, max_length=4096)
    samples: list[OcrSampleV2] = Field(min_length=1, max_length=8192)
    unresolved_annotations: list[UnresolvedOcrAnnotationV2] = Field(default_factory=list, max_length=4096)
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
    def links_and_scope(self):
        sources = {s.id: s for s in self.source_records}
        annotations = {a.id: a for a in self.annotations}
        samples = {s.id: s for s in self.samples}
        if (len(sources) != len(self.source_records) or len(annotations) != len(self.annotations) or
                len(samples) != len(self.samples)):
            raise ValueError('Source, annotation and sample IDs must be unique')
        if (len({s.source.path for s in sources.values()}) != len(sources) or
                len({s.source.sha256 for s in sources.values()}) != len(sources)):
            raise ValueError('Aliased source paths or identical source bytes need one source record')
        all_annotations = [*self.annotations, *self.unresolved_annotations]
        if (len({a.id for a in all_annotations}) != len(all_annotations) or
                len({(a.source_id, a.dimension_id) for a in all_annotations}) != len(all_annotations)):
            raise ValueError('Source/dimension identity may have only one decision')
        if {a.source_id for a in all_annotations} != set(sources):
            raise ValueError('Every source must have decisions and every decision an existing source')
        inherited = set(self.inherited_sample_ids)
        if len(inherited) != len(self.inherited_sample_ids) or not inherited <= set(samples):
            raise ValueError('Inherited sample identities must be unique existing samples')
        dependencies = {p.path: p.sha256 for p in self.dependencies}
        if len(dependencies) != len(self.dependencies):
            raise ValueError('Dependency paths must be unique')
        required_pins = [self.parent_release]
        for source in sources.values():
            required_pins.extend((source.source, source.preparation, source.review,
                                  source.lineage.catalog, source.lineage.split_policy))
        for annotation in all_annotations:
            required_pins.append(annotation.review)
        for pin in required_pins:
            if dependencies.get(pin.path) != pin.sha256:
                raise ValueError('Every source/review/preparation/lineage/parent pin must belong to dependencies')
        links = [sid for a in self.annotations for sid in a.sample_ids]
        if len(set(links)) != len(links) or set(links) != set(samples):
            raise ValueError('Every released sample needs exactly one reviewed annotation line')
        for annotation in self.annotations:
            existing = set(annotation.sample_ids) & inherited
            if existing and existing != set(annotation.sample_ids):
                raise ValueError('An inherited annotation cannot be mixed with newly assigned lines')
            if not existing and annotation.review != sources[annotation.source_id].review:
                raise ValueError('New annotation review must match its source review')
        for annotation in self.unresolved_annotations:
            if annotation.review != sources[annotation.source_id].review:
                raise ValueError('Unresolved decisions must cite their source review')
        crops, all_pins = set(), [*self.dependencies, *required_pins]
        for sample in self.samples:
            annotation = annotations.get(sample.annotation_id)
            if (annotation is None or sample.source_id != annotation.source_id or
                    sample.line_index >= len(annotation.sample_ids) or
                    annotation.sample_ids[sample.line_index] != sample.id or
                    annotation.transcription_lines[sample.line_index] != sample.text):
                raise ValueError('Sample source, line order and text must match the reviewed annotation')
            source = sources[sample.source_id]
            if (sample.parent_region_box_xyxy[2] > source.source_size_px[0] or
                    sample.parent_region_box_xyxy[3] > source.source_size_px[1]):
                raise ValueError('Parent crop exceeds the associated source size')
            if dependencies.get(sample.parent_region_pin.path) != sample.parent_region_pin.sha256:
                raise ValueError('Parent crop evidence must belong to dependencies')
            key = source.source.sha256, tuple(sample.source_box_xyxy)
            if key in crops:
                raise ValueError('A source crop cannot be repeated, even with another orientation')
            crops.add(key)
            all_pins.extend((sample.parent_region_pin, sample.raw_crop_pin, sample.image_pin))
        seen = {}
        for pin in all_pins:
            if pin.path in seen and seen[pin.path] != pin.sha256:
                raise ValueError('A pinned artifact path cannot carry conflicting hashes')
            seen[pin.path] = pin.sha256
        return self


def read_ocr_labels_v2(value):
    """Revalidate a fresh model copy; this is not actual-byte authentication."""
    if isinstance(value, (str, Path)):
        value = read_json_v1(value, max_bytes=16*1024**2)
    if isinstance(value, OcrLabelPackageV2):
        value = value.model_dump(mode='json')
    return OcrLabelPackageV2.model_validate(value)
