"""View-local drawing graph with explicit evidence and versioned v0.1 migration.

Coordinates belong to a view, never implicitly to the sheet. Physical scale,
field review and source provenance are independent of a numeric dimension text.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import DrawingGraph, Issue, Page, Source, TitleBlock
from .tolerances import ToleranceConvention, dimension_tolerances

# These are numerical representation limits, not measurement accuracy bounds.
# They keep affine products and topology predicates within finite double arithmetic.
Number = Annotated[float, Field(ge=-1e12, le=1e12, allow_inf_nan=False)]
Positive = Annotated[float, Field(ge=1e-12, le=1e12, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, le=1e12, allow_inf_nan=False)]
Point = tuple[Number, Number]
Matrix = tuple[tuple[Number, Number, Number], tuple[Number, Number, Number], tuple[Number, Number, Number]]
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class SourceRegion(Model):
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    coordinate_space: Literal['source_pixel', 'paper_mm']
    box: tuple[Number, Number, Number, Number]

    @model_validator(mode='after')
    def ordered(self):
        x0, y0, x1, y1 = self.box
        if not (x0 < x1 and y0 < y1):
            raise ValueError('Source region must have positive extent')
        if self.coordinate_space == 'source_pixel' and min(self.box) < 0:
            raise ValueError('Source pixel bounds cannot be negative')
        return self


class Uncertainty(Model):
    bound: Nonnegative
    unit: Literal['source_pixel', 'view_unit', 'model_mm', 'degree']
    basis: str = Field(min_length=1)
    statistical_confidence: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None = None


class Evidence(Model):
    source_ref: str = Field(min_length=1)
    method: Literal['ocr', 'geometry_fit', 'derived', 'human_verified', 'migrated_v1']
    source_region: SourceRegion | None = None
    uncertainty: Uncertainty | None = None
    status: Literal['unknown', 'observed', 'verified'] = 'observed'
    review_ref: str | None = None

    @model_validator(mode='after')
    def reviewed(self):
        if self.status == 'verified' and not self.review_ref:
            raise ValueError('Verified evidence requires an explicit review reference')
        if self.method == 'migrated_v1' and self.status == 'verified':
            raise ValueError('Migration cannot create verified evidence')
        return self


class LayerSpec(Model):
    name: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    role: Literal['main', 'thin', 'axis', 'dimension', 'text', 'hatch', 'symbol', 'hidden', 'other']
    line_type: Literal['CONTINUOUS', 'CENTER', 'HIDDEN'] = 'CONTINUOUS'
    color: int = Field(default=7, ge=1, le=255)
    lineweight: int = Field(default=25, ge=0, le=211)

    @model_validator(mode='after')
    def supported_weight(self):
        if self.lineweight not in {0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158, 200, 211}:
            raise ValueError('Layer lineweight is not a supported DXF value')
        return self


def default_layers():
    return [LayerSpec(name=name, role=role, line_type=lt, lineweight=weight)
            for name, role, lt, weight in [('MAIN', 'main', 'CONTINUOUS', 50),
                ('THIN', 'thin', 'CONTINUOUS', 25), ('AXIS', 'axis', 'CENTER', 25),
                ('DIMS', 'dimension', 'CONTINUOUS', 25), ('TEXT', 'text', 'CONTINUOUS', 25)]]


class View(Model):
    id: str = Field(min_length=1)
    name: str | None = None
    evidence: Evidence
    coordinate_space: Literal['pixel', 'paper_mm', 'model_mm'] = 'paper_mm'
    # All local frames use X right, Y up; this optional transform retains pixel crop/flip provenance.
    source_to_view: Matrix | None = None
    # Mapping of view-local coordinates to the original sheet, independent of scale calibration.
    view_to_sheet: Matrix = IDENTITY
    # Physical model millimetres per local unit, X and Y separately only with evidence.
    model_scale_xy: tuple[Positive, Positive] | None = None
    model_origin: Point = (0., 0.)
    model_rotation_deg: Number = 0
    physical_scale_confirmed: bool = False
    scale_evidence_refs: list[str] = Field(default_factory=list)
    deformation_confirmed: bool = False
    deformation_evidence_ref: str | None = None

    @model_validator(mode='after')
    def transforms(self):
        a = self.view_to_sheet
        if a[2] != (0., 0., 1.) or abs(a[0][0]*a[1][1]-a[0][1]*a[1][0]) < 1e-12:
            raise ValueError('View-to-sheet must be a finite invertible affine transform')
        if self.source_to_view is not None:
            s = self.source_to_view
            if s[2] != (0., 0., 1.) or abs(s[0][0]*s[1][1]-s[0][1]*s[1][0]) < 1e-12:
                raise ValueError('Source-to-view must be a finite invertible affine transform')
        if self.physical_scale_confirmed and (self.model_scale_xy is None or not self.scale_evidence_refs
                                               or any(not s.strip() for s in self.scale_evidence_refs)):
            raise ValueError('Physical scale needs explicit factors and evidence')
        if self.deformation_confirmed and not self.deformation_evidence_ref:
            raise ValueError('Confirmed deformation requires an evidence reference')
        if self.model_scale_xy is not None:
            sx, sy = self.model_scale_xy
            if not math.isclose(sx, sy, rel_tol=1e-10, abs_tol=1e-12) and not (
                    self.deformation_confirmed and self.deformation_evidence_ref):
                raise ValueError('Anisotropic scale requires confirmed deformation evidence')
            if self.coordinate_space == 'model_mm' and (sx != 1 or sy != 1):
                raise ValueError('Already physical model coordinates cannot be rescaled')
        return self

    def model_matrix(self) -> Matrix:
        if not self.physical_scale_confirmed or self.model_scale_xy is None:
            raise ValueError('Physical transform is not confirmed')
        sx, sy = self.model_scale_xy
        angle = math.radians(self.model_rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        x, y = self.model_origin
        return ((c*sx, -s*sy, x), (s*sx, c*sy, y), (0., 0., 1.))


class Entity(Model):
    id: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    evidence: Evidence
    layer: str = 'MAIN'
    status: Literal['proposed', 'accepted', 'rejected'] = 'proposed'


class LineV2(Entity):
    kind: Literal['line'] = 'line'
    start: Point
    end: Point

    @model_validator(mode='after')
    def nonzero(self):
        if math.dist(self.start, self.end) <= 1e-9:
            raise ValueError('Zero-length line')
        return self


class CircleV2(Entity):
    kind: Literal['circle'] = 'circle'
    center: Point
    radius: Positive


class ArcV2(Entity):
    kind: Literal['arc'] = 'arc'
    center: Point
    radius: Positive
    start_angle_deg: Number
    end_angle_deg: Number

    @model_validator(mode='after')
    def nonzero(self):
        if (self.end_angle_deg-self.start_angle_deg) % 360 < 1e-9:
            raise ValueError('An ARC needs a nonzero sweep smaller than a full circle')
        return self


class Vertex(Model):
    point: Point
    bulge: Number = 0

    @model_validator(mode='after')
    def resolvable_bulge(self):
        if self.bulge and not 1e-12 <= abs(self.bulge) <= 1e6:
            raise ValueError('Bulge is outside the supported numerical curvature range')
        return self


class PolylineV2(Entity):
    kind: Literal['polyline'] = 'polyline'
    vertices: list[Vertex] = Field(min_length=2, max_length=10000)
    closed: bool = False

    @model_validator(mode='after')
    def segments(self):
        ends = self.vertices[1:]+(self.vertices[:1] if self.closed else [])
        if any(math.dist(a.point, b.point) <= 1e-9 for a, b in zip(self.vertices, ends)):
            raise ValueError('Polyline has a zero-length segment')
        if not self.closed and self.vertices[-1].bulge != 0:
            raise ValueError('Last vertex of an open polyline cannot carry a segment bulge')
        if self.closed and len(self.vertices) < 3 and all(v.bulge == 0 for v in self.vertices):
            raise ValueError('A straight closed polyline needs at least three vertices')
        return self


GeometryV2 = Annotated[LineV2 | CircleV2 | ArcV2 | PolylineV2, Field(discriminator='kind')]


class TextV2(Entity):
    layer: str = 'TEXT'
    raw: str
    position: Point
    height: Positive
    rotation_deg: Number = 0
    alternatives: list[str] = Field(default_factory=list)


class DimensionV2(Entity):
    layer: str = 'DIMS'
    kind: Literal['linear', 'aligned', 'radius', 'diameter', 'longitudinal_diameter', 'angular']
    nominal: Positive
    raw_text: str
    # Angular: vertex, ray 1, ray 2 (CCW). Radius/diameter: center, perimeter.
    # Other kinds: measured endpoints, with projection for linear/longitudinal.
    points: list[Point] = Field(min_length=2, max_length=3)
    placement: Point
    measured_entity_ids: list[str] = Field(min_length=1)
    angle_deg: Number = 0
    prefix: str = ''
    suffix: str = ''
    tolerance_plus: Number | None = None
    tolerance_minus: Number | None = None
    # New objects use signed deviations. read_graph_v2 explicitly tags unmarked
    # historical JSON as magnitudes, so existing files never change meaning.
    tolerance_convention: ToleranceConvention = 'signed_deviations'
    # Model mm per printed paper mm controls graphical size, not numeric measurement.
    annotation_scale: Positive | None = None

    @model_validator(mode='after')
    def fields(self):
        if len(self.points) != (3 if self.kind == 'angular' else 2):
            raise ValueError('Dimension kind has the wrong number of defining points')
        if self.kind == 'angular' and self.nominal >= 360:
            raise ValueError('Angular nominal must be below 360 degrees')
        if any(token in self.prefix+self.suffix for token in ('<', '>', '\\', '{', '}', '\n', '\r')):
            raise ValueError('Affixes must be literal single-line text, without override placeholders or commands')
        if len(set(self.measured_entity_ids)) != len(self.measured_entity_ids):
            raise ValueError('Dimension geometry references must be distinct')
        dimension_tolerances(self)
        return self


class ContourEdge(Model):
    primitive_id: str
    reversed: bool = False


class Contour(Entity):
    edges: list[ContourEdge] = Field(min_length=1, max_length=1000)
    closed: bool
    role: Literal['outer', 'hole', 'open', 'unknown'] = 'unknown'
    parent_id: str | None = None

    @model_validator(mode='after')
    def boundary(self):
        if self.role in ('outer', 'hole') and not self.closed:
            raise ValueError('A boundary or hole must be a closed contour')
        if self.role == 'open' and self.closed:
            raise ValueError('An open contour cannot be marked closed')
        if (self.role == 'hole') != (self.parent_id is not None):
            raise ValueError('Only a hole has an explicit parent contour')
        ids = [edge.primitive_id for edge in self.edges]
        if len(ids) != len(set(ids)):
            raise ValueError('A contour cannot traverse one primitive twice')
        return self


class Hatch(Entity):
    layer: str = 'THIN'
    outer_contour_id: str
    hole_contour_ids: list[str] = Field(default_factory=list, max_length=64)
    pattern: Literal['SOLID', 'ANSI31'] = 'ANSI31'
    angle_deg: Number = 45
    spacing: Positive = 2


class ConstraintV2(Entity):
    kind: Literal['distance', 'projected_distance', 'diameter', 'radius', 'parallel', 'perpendicular',
                  'tangent', 'concentric', 'symmetric', 'equal_radius', 'coincident', 'fixed']
    entity_ids: list[str] = Field(min_length=1)
    value: Number | None = None
    parameters: dict = Field(default_factory=dict)


class DrawingGraphV2(Model):
    schema_version: Literal['0.2'] = '0.2'
    source: Source
    page: Page
    views: list[View] = Field(min_length=1, max_length=1000)
    layers: list[LayerSpec] = Field(default_factory=default_layers)
    primitives: list[GeometryV2] = Field(default_factory=list)
    texts: list[TextV2] = Field(default_factory=list)
    dimensions: list[DimensionV2] = Field(default_factory=list)
    contours: list[Contour] = Field(default_factory=list)
    hatches: list[Hatch] = Field(default_factory=list)
    constraints: list[ConstraintV2] = Field(default_factory=list)
    title_block: TitleBlock = Field(default_factory=TitleBlock)
    issues: list[Issue] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def references(self):
        if max(self.page.width_mm, self.page.height_mm) > 1e12 or min(self.page.width_mm, self.page.height_mm) < 1e-12:
            raise ValueError('Sheet size exceeds the supported numerical representation')
        view_ids = {v.id for v in self.views}
        if len(view_ids) != len(self.views):
            raise ValueError('View IDs must be unique')
        layers = {layer.name for layer in self.layers}
        if len(layers) != len(self.layers) or not {'MAIN', 'THIN', 'AXIS', 'DIMS', 'TEXT'} <= layers:
            raise ValueError('Unique layers must include the five required base layers')
        items = [*self.primitives, *self.texts, *self.dimensions, *self.contours, *self.hatches, *self.constraints]
        if len({item.id for item in items}) != len(items):
            raise ValueError('Entity IDs must be unique across the graph')
        for item in items:
            if item.view_id not in view_ids or item.layer not in layers:
                raise ValueError('Every entity needs an existing view and explicitly declared layer')
        for item in [*self.views, *items]:
            region = item.evidence.source_region
            if region is not None and region.source_sha256 != self.source.sha256:
                raise ValueError('Source regions must identify the graph source, not an unrelated image')
        geometries = {p.id: p for p in self.primitives}
        contours = {c.id: c for c in self.contours}
        for dimension in self.dimensions:
            self._same_view(dimension, dimension.measured_entity_ids, geometries)
        for constraint in self.constraints:
            self._same_view(constraint, constraint.entity_ids, geometries)
        for contour in self.contours:
            self._same_view(contour, [e.primitive_id for e in contour.edges], geometries)
            if contour.parent_id:
                self._same_view(contour, [contour.parent_id], contours)
                if contours[contour.parent_id].role != 'outer':
                    raise ValueError('A hole must reference an outer contour')
        for hatch in self.hatches:
            self._same_view(hatch, [hatch.outer_contour_id, *hatch.hole_contour_ids], contours)
            if contours[hatch.outer_contour_id].role != 'outer':
                raise ValueError('HATCH requires an outer boundary')
            if len(set(hatch.hole_contour_ids)) != len(hatch.hole_contour_ids):
                raise ValueError('Repeated HATCH hole')
            if any(contours[h].role != 'hole' or contours[h].parent_id != hatch.outer_contour_id
                   for h in hatch.hole_contour_ids):
                raise ValueError('HATCH holes must belong to its outer contour')
        return self

    @staticmethod
    def _same_view(owner, refs, targets):
        if any(ref not in targets or targets[ref].view_id != owner.view_id for ref in refs):
            raise ValueError('References must exist and remain within the owning view')


def migrate_v1(graph: DrawingGraph | dict) -> DrawingGraphV2:
    """Preserve the original snapshot and claims without inventing verified fields."""
    old = graph if isinstance(graph, DrawingGraph) else DrawingGraph.model_validate(graph)
    snapshot = deepcopy(graph) if isinstance(graph, dict) else old.model_dump(mode='json')
    view_id = 'legacy-sheet'

    def evidence(item):
        uncertainty = getattr(item, 'uncertainty_mm', None)
        return Evidence(source_ref=item.source_ref, method='migrated_v1',
            uncertainty=Uncertainty(bound=uncertainty, unit='view_unit', basis='legacy uncertainty_mm') if uncertainty is not None else None)

    layer_map = {x.name: x for x in default_layers()}
    primitives = []
    for primitive in old.primitives:
        data = primitive.model_dump()
        line_type = data.pop('line_type')
        layer = data['layer']
        if layer_map[layer].line_type != line_type:
            layer = layer+'_'+line_type
            layer_map.setdefault(layer, LayerSpec(name=layer, role='hidden' if line_type == 'HIDDEN' else 'other', line_type=line_type))
        data.update(view_id=view_id, evidence=evidence(primitive), layer=layer)
        data.pop('source_ref')
        data.pop('uncertainty_mm')
        if primitive.kind == 'polyline':
            data['vertices'] = [{'point': p, 'bulge': 0} for p in data.pop('points')]
        primitives.append(data)
    texts = [dict(id=t.id, view_id=view_id, evidence=evidence(t), raw=t.raw, position=t.position,
                  height=t.height_mm, rotation_deg=t.rotation_deg, alternatives=t.alternatives) for t in old.texts]
    dimensions = []
    for dimension in old.dimensions:
        data = dimension.model_dump()
        data.pop('source_ref')
        data.update(view_id=view_id, evidence=evidence(dimension),
                    tolerance_convention='signed_deviations')
        dimensions.append(data)
    constraints = []
    for constraint in old.constraints:
        data = constraint.model_dump()
        data.pop('source_ref')
        # A legacy conflict remains an issue and cannot become an accepted constraint.
        if data['status'] == 'conflict':
            data['status'] = 'proposed'
        data.update(view_id=view_id, evidence=evidence(constraint))
        constraints.append(data)
    factor = old.page.model_per_paper if old.coordinate_space == 'paper_mm' else 1.
    return DrawingGraphV2(source=old.source, page=old.page,
        views=[View(id=view_id, name='Unsegmented legacy sheet',
                    evidence=Evidence(source_ref=old.source.path, method='migrated_v1'),
                    coordinate_space=old.coordinate_space, model_scale_xy=(factor, factor) if factor else None)],
        layers=list(layer_map.values()), primitives=primitives, texts=texts, dimensions=dimensions, constraints=constraints,
        title_block=old.title_block, issues=[*old.issues,
            *[Issue(code='legacy_constraint_conflict', message='An unresolved v0.1 constraint conflict is retained.',
                    entity_ids=[c.id]) for c in old.constraints if c.status == 'conflict'],
            Issue(code='migration_requires_field_review',
            message='Legacy source regions, view membership and physical scale remain unverified.')],
        provenance={'migration': {'from_version': '0.1', 'original_graph': snapshot,
                                 'approvals_created': False}, 'legacy_provenance': deepcopy(old.provenance)})


def read_graph_v2(value: DrawingGraphV2 | DrawingGraph | dict | str | Path) -> DrawingGraphV2:
    """Read either graph version, always returning a revalidated independent v2 model."""
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding='utf-8'))
    if isinstance(value, DrawingGraph):
        return migrate_v1(value)
    if isinstance(value, DrawingGraphV2):
        value = value.model_dump(mode='json')
    if not isinstance(value, dict):
        raise ValueError('Expected a graph object or a JSON file')
    if value.get('schema_version') == '0.1':
        return migrate_v1(value)
    if value.get('schema_version') == '0.2':
        value = deepcopy(value)
        dimensions = value.get('dimensions', [])
        if isinstance(dimensions, list):
            for dimension in dimensions:
                if isinstance(dimension, dict):
                    dimension.setdefault('tolerance_convention', 'legacy_magnitudes')
    return DrawingGraphV2.model_validate(value)
