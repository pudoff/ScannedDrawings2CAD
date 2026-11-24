"""Export accepted graph data to editable DXF and verify the saved file.

The graph carries measured geometry; nominal text never changes that geometry.
Native dimensions are non-associative: their graph references are validated here,
but CAD applications will not automatically follow subsequent geometry edits.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import ezdxf

from .schema import Arc, Circle, Dimension, DrawingGraph, Geometry, Line, Polyline
from .tolerances import dimension_tolerances


# Numerical consistency, not a manufacturing or recognition tolerance.
ABS_TOL_MM = 1e-4
REL_TOL = 1e-6


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ABS_TOL_MM, rel_tol=REL_TOL)


def _point(point, factor: float) -> tuple[float, float]:
    return point[0] * factor, point[1] * factor


def _on_segment(point, start, end, tolerance: float) -> bool:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_squared = vx * vx + vy * vy
    if length_squared == 0:
        return math.dist(point, start) <= tolerance
    projection = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = start[0] + projection * vx, start[1] + projection * vy
    return math.dist(point, nearest) <= tolerance


def _on_arc(point, arc: Arc, tolerance: float) -> bool:
    angle = math.degrees(math.atan2(point[1] - arc.center[1], point[0] - arc.center[0]))
    sweep = (arc.end_angle_deg - arc.start_angle_deg) % 360
    delta = (angle - arc.start_angle_deg) % 360
    angle_tolerance = math.degrees(tolerance / arc.radius)
    return delta <= sweep + angle_tolerance or 360 - delta <= angle_tolerance


def _attached(point, geometry: Geometry, tolerance: float, *, allow_center: bool = True) -> bool:
    if isinstance(geometry, Line):
        return _on_segment(point, geometry.start, geometry.end, tolerance)
    if isinstance(geometry, Polyline):
        vertices = [*geometry.points, geometry.points[0]] if geometry.closed else geometry.points
        return any(_on_segment(point, a, b, tolerance) for a, b in zip(vertices, vertices[1:]))
    if isinstance(geometry, (Circle, Arc)):
        distance = math.dist(point, geometry.center)
        if allow_center and distance <= tolerance:
            return True
        if abs(distance - geometry.radius) > tolerance:
            return False
        return not isinstance(geometry, Arc) or _on_arc(point, geometry, tolerance)
    return False


def _dimension_error(dim: Dimension, geometries: dict, factor: float) -> tuple[str | None, float]:
    a, b = dim.points
    references = [geometries[entity_id] for entity_id in dim.measured_entity_ids]
    tolerance = ABS_TOL_MM / factor
    if dim.kind == "linear":
        angle = math.radians(dim.angle_deg)
        measurement = abs((b[0] - a[0]) * math.cos(angle) + (b[1] - a[1]) * math.sin(angle)) * factor
        attached = all(any(_attached(p, g, tolerance) for g in references) for p in (a, b))
        attached = attached and all(any(_attached(p, g, tolerance) for p in (a, b)) for g in references)
    else:
        measurement = math.dist(a, b) * factor * (2 if dim.kind == "diameter" else 1)
        attached = all(
            isinstance(g, (Circle, Arc))
            and math.dist(a, g.center) <= tolerance
            and _attached(b, g, tolerance, allow_center=False)
            for g in references
        )
    if not attached:
        return "dimension_anchor_mismatch", measurement
    if measurement <= ABS_TOL_MM or not _close(measurement, dim.nominal):
        return "dimension_nominal_mismatch", measurement
    return None, measurement


def _audit_entries(entries) -> list[dict]:
    return [{"code": int(entry.code), "message": entry.message} for entry in entries]


def export_dxf(graph: DrawingGraph, output: Path) -> dict:
    """Write AC1021/UTF-8 DXF and return a JSON-serializable verification report.

    Known paper scale converts all graph coordinates to physical model mm.
    Unknown paper scale preserves observations at paper size and rejects physical
    dimensions. Layout coordinates are always paper mm; viewport scale is 1/S.
    Issues in source data yield ``needs_review`` even if the DXF is structurally
    valid. Failed I/O, roundtrip mismatch, or uncorrected audit errors yield
    ``failed``. Source graph and original files are never modified.
    """
    output = Path(output)
    scale = graph.page.model_per_paper
    factor = scale if graph.coordinate_space == "paper_mm" and scale is not None else 1.0
    viewport_scale = scale if scale is not None else 1.0
    observation_only = graph.coordinate_space == "paper_mm" and scale is None
    report = {
        "status": "needs_review",
        "output_path": str(output),
        "dxf_version": "AC1021",
        "encoding": "utf-8",
        "coordinate_conventions": {
            "input_space": graph.coordinate_space,
            "modelspace": "paper_observation_mm" if observation_only else "model_mm",
            "axes": "X right, Y up",
            "input_to_model_factor": factor,
            "layout_units": "mm",
            "model_per_paper": scale,
            "viewport_paper_per_model": 1.0 / viewport_scale,
            "viewport_scale_confirmed": scale is not None,
            "unknown_scale_policy": "preserve coordinates; viewport 1:1 is an unconfirmed observation",
            "dimensions_associative": False,
        },
        "counts": {"primitives": 0, "texts": 0, "dimensions": 0, "dimensions_skipped": 0},
        "dimensions": [],
        "audit": {"readable": False, "errors": [], "fixes": [], "cad_application_checked": False},
        "errors": [],
        "issues": [issue.model_dump() for issue in graph.review_issues()],
    }

    def issue(code, message, entity_ids=None, severity="warning"):
        item = {"code": code, "message": message, "severity": severity, "entity_ids": entity_ids or []}
        report["issues"].append(item)
        if severity == "error":
            report["errors"].append(item.copy())

    try:
        if output.resolve() == Path(graph.source.path).resolve():
            raise ValueError("Output path must differ from the original source path")
        doc = ezdxf.new("R2007", setup=True)
        doc.units = ezdxf.units.MM
        doc.header["$MEASUREMENT"] = 1
        doc.header["$LTSCALE"] = viewport_scale
        doc.header["$DIMASSOC"] = 1  # editable, non-associative DIMENSION entities
        for name, weight, linetype in (
            ("MAIN", 50, "CONTINUOUS"), ("THIN", 25, "CONTINUOUS"),
            ("AXIS", 25, "CENTER"), ("DIMS", 25, "CONTINUOUS"), ("TEXT", 25, "CONTINUOUS"),
        ):
            if linetype == "CENTER" and linetype not in doc.linetypes:
                doc.linetypes.new("CENTER", dxfattribs={"pattern": [20.0, 12.0, -3.0, 2.0, -3.0]})
            doc.layers.new(name, dxfattribs={"lineweight": weight, "linetype": linetype})
        if "HIDDEN" not in doc.linetypes:
            doc.linetypes.new("HIDDEN", dxfattribs={"pattern": [6.0, 3.0, -3.0]})
        model = doc.modelspace()
        expected = Counter()
        for primitive in graph.primitives:
            attributes = {"layer": primitive.layer, "linetype": primitive.line_type}
            if isinstance(primitive, Line):
                entity = model.add_line(_point(primitive.start, factor), _point(primitive.end, factor), dxfattribs=attributes)
            elif isinstance(primitive, Circle):
                entity = model.add_circle(_point(primitive.center, factor), primitive.radius * factor, dxfattribs=attributes)
            elif isinstance(primitive, Arc):
                entity = model.add_arc(_point(primitive.center, factor), primitive.radius * factor,
                                       primitive.start_angle_deg, primitive.end_angle_deg, dxfattribs=attributes)
            else:
                entity = model.add_lwpolyline([_point(p, factor) for p in primitive.points],
                                             close=primitive.closed, dxfattribs=attributes)
            expected[entity.dxftype()] += 1
            report["counts"]["primitives"] += 1

        for text in graph.texts:
            model.add_text(text.raw, dxfattribs={"layer": "TEXT", "insert": _point(text.position, factor),
                                                "height": text.height_mm * factor, "rotation": text.rotation_deg})
            expected["TEXT"] += 1
            report["counts"]["texts"] += 1

        geometries = {primitive.id: primitive for primitive in graph.primitives}
        for dim in graph.dimensions:
            if dim.status != "accepted":
                report["counts"]["dimensions_skipped"] += 1
                continue
            if observation_only:
                issue("dimension_scale_unknown", "Physical dimension cannot be verified while paper scale is unknown.", [dim.id])
                report["counts"]["dimensions_skipped"] += 1
                continue
            error, measurement = _dimension_error(dim, geometries, factor)
            if error:
                issue(error, f"Dimension {dim.id}: anchors must match referenced geometry and nominal "
                      f"{dim.nominal:g} mm must match measurement {measurement:g} mm; dimension omitted.",
                      [dim.id, *dim.measured_entity_ids], "error")
                report["counts"]["dimensions_skipped"] += 1
                continue
            # dimlfac stays 1: never disguise geometry errors by changing displayed values.
            override = {"dimlfac": 1.0, "dimscale": viewport_scale, "dimtxt": 2.5, "dimasz": 2.5, "dimdec": 6}
            try:
                deviations = dimension_tolerances(dim)
            except ValueError as error:
                issue('dimension_tolerance_invalid', str(error), [dim.id], 'error')
                report['counts']['dimensions_skipped'] += 1
                continue
            if deviations is not None:
                override.update(deviations.dxf_style())
            a, b = (_point(p, factor) for p in dim.points)
            placement = _point(dim.placement, factor)
            if dim.kind == "linear":
                native = model.add_linear_dim(base=placement, p1=a, p2=b, angle=dim.angle_deg,
                                               override=override, dxfattribs={"layer": "DIMS"})
            else:
                create = model.add_radius_dim if dim.kind == "radius" else model.add_diameter_dim
                native = create(center=a, mpoint=b, override=override, dxfattribs={"layer": "DIMS"})
                native.user_location_override(placement)
            native.render()
            report["dimensions"].append({"id": dim.id, "handle": native.dimension.dxf.handle,
                                         "kind": dim.kind, "nominal_mm": dim.nominal,
                                         "measurement_mm": measurement,
                                         "measured_entity_ids": list(dim.measured_entity_ids)})
            expected["DIMENSION"] += 1
            report["counts"]["dimensions"] += 1

        sheet = doc.layouts.get("Layout1")
        doc.layouts.rename("Layout1", "Sheet")
        width, height = graph.page.width_mm, graph.page.height_mm
        sheet.page_setup(size=(width, height), margins=(0, 0, 0, 0), units="mm", scale=(1, 1))
        # The complete original page is shown, without fit-to-extents recentering.
        sheet.add_viewport(center=(width / 2, height / 2), size=(width, height),
                           view_center_point=(width * viewport_scale / 2, height * viewport_scale / 2),
                           view_height=height * viewport_scale)
        output.parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(output, encoding="utf-8")
        saved = ezdxf.readfile(output)
        auditor = saved.audit()
        report["audit"].update(readable=True, errors=_audit_entries(auditor.errors), fixes=_audit_entries(auditor.fixes))
        actual = Counter(entity.dxftype() for entity in saved.modelspace())
        report["counts"]["modelspace_by_type"] = dict(actual)
        if actual != expected:
            issue("roundtrip_entity_mismatch", f"Expected {dict(expected)}, read back {dict(actual)}.", severity="error")
            report["status"] = "failed"
            return report
        for dim_report in report["dimensions"]:
            native = saved.entitydb[dim_report["handle"]]
            measured = float(native.get_measurement())
            dim_report["roundtrip_measurement_mm"] = measured
            if not _close(measured, dim_report["nominal_mm"]) or native.dxf.text not in ("", "<>"):
                issue("roundtrip_dimension_mismatch", "Saved DIMENSION measurement differs from its nominal.",
                      [dim_report["id"]], "error")
                report["status"] = "failed"
                return report
        if auditor.errors:
            issue("dxf_audit_errors", "DXF audit found unresolved structural errors.", severity="error")
            report["status"] = "failed"
        else:
            if auditor.fixes:
                issue("dxf_audit_fixes", "Reading the saved DXF required audit repairs; inspect the file.")
            report["status"] = "needs_review" if any(i["severity"] != "info" for i in report["issues"]) else "succeeded"
    except (OSError, ValueError, TypeError, ezdxf.DXFError) as error:
        issue("dxf_export_failed", f"{type(error).__name__}: {error}", severity="error")
        report["status"] = "failed"
    return report
