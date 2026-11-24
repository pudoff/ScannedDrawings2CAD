"""Versioned observations and accepted geometry, with explicit coordinate units."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tolerances import dimension_tolerances

Number = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Point = tuple[Number, Number]
Layer = Literal["MAIN", "THIN", "AXIS", "DIMS", "TEXT"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Source(Model):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_index: int = Field(ge=0)
    method: str


class Issue(Model):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    entity_ids: list[str] = Field(default_factory=list)


class Page(Model):
    width_mm: Positive
    height_mm: Positive
    size_source: Literal["pdf_media", "image_dpi", "user", "unknown"]
    kind: Literal["vector", "raster", "mixed", "empty"]
    # Physical model length / printed paper length; 1:5 -> 5. None is unknown.
    model_per_paper: Positive | None = None
    scale_source: str | None = None
    rotation_deg: Number = 0

    @model_validator(mode="after")
    def scale_has_evidence(self):
        if self.model_per_paper is not None and not self.scale_source:
            raise ValueError("Known scale requires scale_source")
        return self


class Primitive(Model):
    id: str
    layer: Layer = "MAIN"
    line_type: Literal["CONTINUOUS", "CENTER", "HIDDEN"] = "CONTINUOUS"
    source_ref: str
    uncertainty_mm: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class Line(Primitive):
    kind: Literal["line"] = "line"
    start: Point
    end: Point

    @model_validator(mode="after")
    def nonzero(self):
        if math.dist(self.start, self.end) < 1e-9:
            raise ValueError("Zero-length line")
        return self


class Circle(Primitive):
    kind: Literal["circle"] = "circle"
    center: Point
    radius: Positive


class Arc(Primitive):
    kind: Literal["arc"] = "arc"
    center: Point
    radius: Positive
    start_angle_deg: Number
    end_angle_deg: Number


class Polyline(Primitive):
    kind: Literal["polyline"] = "polyline"
    points: list[Point] = Field(min_length=2)
    closed: bool = False


Geometry = Annotated[Line | Circle | Arc | Polyline, Field(discriminator="kind")]


class Text(Model):
    id: str
    raw: str
    position: Point
    height_mm: Positive
    rotation_deg: Number = 0
    source_ref: str
    alternatives: list[str] = Field(default_factory=list)


class Dimension(Model):
    id: str
    kind: Literal["linear", "diameter", "radius"]
    nominal: Positive
    raw_text: str
    # Linear: two measured points. Radius/diameter: center and point on circle.
    points: list[Point] = Field(min_length=2, max_length=2)
    placement: Point
    measured_entity_ids: list[str] = Field(min_length=1)
    source_ref: str
    status: Literal["proposed", "accepted", "rejected"] = "proposed"
    angle_deg: Number = 0
    tolerance_plus: Number | None = Field(default=None, description="Signed upper deviation from nominal")
    tolerance_minus: Number | None = Field(default=None, description="Signed lower deviation from nominal")

    @model_validator(mode="after")
    def tolerance_interval(self):
        dimension_tolerances(self)
        return self


class Constraint(Model):
    id: str
    kind: Literal["distance", "diameter", "radius", "parallel", "perpendicular"]
    entity_ids: list[str] = Field(min_length=1)
    value: Number | None = None
    source_ref: str
    status: Literal["proposed", "accepted", "conflict"] = "proposed"


class TitleField(Model):
    value: str | None = None
    source_ref: str | None = None
    status: Literal["unknown", "observed", "verified"] = "unknown"

    @model_validator(mode="after")
    def known_value_has_evidence(self):
        if self.status != "unknown" and (not self.value or not self.value.strip() or not self.source_ref):
            raise ValueError("Observed or verified title field requires a value and source_ref")
        return self


class TitleBlock(Model):
    name: TitleField = Field(default_factory=TitleField)
    document_number: TitleField = Field(default_factory=TitleField)
    material: TitleField = Field(default_factory=TitleField)
    authors: TitleField = Field(default_factory=TitleField)


class DrawingGraph(Model):
    schema_version: Literal["0.1"] = "0.1"
    source: Source
    page: Page
    # X right, Y up. All distances mm. PDF native points are retained in provenance.
    coordinate_space: Literal["paper_mm", "model_mm"] = "paper_mm"
    primitives: list[Geometry] = Field(default_factory=list)
    texts: list[Text] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    title_block: TitleBlock = Field(default_factory=TitleBlock)
    issues: list[Issue] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_exist(self):
        entities = [*self.primitives, *self.texts, *self.dimensions, *self.constraints]
        ids = [entity.id for entity in entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Entity IDs must be unique")
        primitive_ids = {entity.id for entity in self.primitives}
        for dim in self.dimensions:
            if not set(dim.measured_entity_ids) <= primitive_ids:
                raise ValueError(f"Dimension {dim.id} references missing geometry")
        for constraint in self.constraints:
            if not set(constraint.entity_ids) <= primitive_ids:
                raise ValueError(f"Constraint {constraint.id} references missing geometry")
        return self

    def review_issues(self) -> list[Issue]:
        result = list(self.issues)
        if self.page.model_per_paper is None:
            result.append(Issue(code="scale_unknown", message="Масштаб детали не подтверждён."))
        if not self.primitives:
            result.append(Issue(code="geometry_missing", message="Геометрия не восстановлена."))
        if not any(d.status == "accepted" for d in self.dimensions):
            result.append(Issue(code="dimensions_unresolved", message="Нет подтверждённых размерных связей."))
        if any(field.status != "verified" for field in self.title_block.__dict__.values()):
            result.append(Issue(code="title_block_unverified", message="Обязательные поля штампа не проверены."))
        if any(d.status == "proposed" for d in self.dimensions):
            result.append(Issue(code="dimension_proposals", message="Есть непроверенные размерные связи."))
        if any(c.status != "accepted" for c in self.constraints):
            result.append(Issue(code="constraints_unresolved", message="Есть непроверенные или противоречивые ограничения."))
        return result
