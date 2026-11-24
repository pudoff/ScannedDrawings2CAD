"""Read-only CAD evidence for pair review, never an automatic training label.

Measurements and definition points are native DXF observations. A rendered
dimension block can be stale; agreeing text is not proof of attachment to a part.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import re

import ezdxf
from ezdxf.math import Matrix44, Vec3
from ezdxf.units import unit_name

MAX_FILE_BYTES = 50 * 1024**2
MAX_ENTITIES = 200_000
MAX_EXPANDED_ENTITIES = 30_000
MAX_BLOCK_DEPTH = 16
MAX_BLOCK_TEXTS = 1000
DIMENSION_TYPES = {0: "linear", 1: "aligned", 2: "angular_two_lines", 3: "diameter",
                   4: "radius", 5: "angular_three_points", 6: "ordinate"}
STYLE_FIELDS = ("dimlfac", "dimscale", "dimdec", "dimrnd", "dimlunit", "dimzin",
                "dimpost", "dimalt", "dimtol", "dimlim", "dimtp", "dimtm", "dimtxsty")


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (Vec3, tuple, list)):
        return [_value(v) for v in value]
    return str(value)


def _matrix(matrix):
    # ezdxf matrices use row vectors: [x y z 1] @ matrix.
    return [[float(v) for v in matrix.get_row(i)] for i in range(4)]


def _simple_number(text):
    # Do not infer a nominal from tolerances, threads, fractions or multiline text.
    text = text.strip().replace("%%c", "Ø").replace("%%C", "Ø")
    match = re.fullmatch(r"(?:[Øø⌀Rr]\s*)?([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))", text)
    if match:
        number = float(match[1].replace(",", "."))
        return number if math.isfinite(number) else None
    return None


def _block_texts(block, document):
    """Inspect stored display text, including nested blocks, without regenerating."""
    texts, issues = [], []
    visited = 0

    def walk(layout, ancestors):
        nonlocal visited
        if layout.name in ancestors:
            issues.append("dimension_display_block_cycle")
            return
        if len(ancestors) >= MAX_BLOCK_DEPTH:
            issues.append("dimension_display_depth_limit")
            return
        for entity in layout:
            visited += 1
            if visited > MAX_EXPANDED_ENTITIES or len(texts) >= MAX_BLOCK_TEXTS:
                issues.append("dimension_display_budget_exceeded")
                return
            if entity.dxftype() in ("TEXT", "MTEXT", "ATTRIB"):
                texts.append({"handle": entity.dxf.handle, "type": entity.dxftype(),
                              "block": layout.name, "text": entity.plain_text(),
                              "raw_text": entity.text if entity.dxftype() == "MTEXT" else entity.dxf.text})
            elif entity.dxftype() == "INSERT":
                nested = document.blocks.get(entity.dxf.name)
                if nested is None:
                    issues.append("dimension_display_missing_block")
                else:
                    walk(nested, (*ancestors, layout.name))

    if block is not None:
        walk(block, ())
    return texts, sorted(set(issues))


def _dimension(entity, document, owner, insert_path, transform):
    kind = entity.dimtype
    standard_dimension = entity.dxftype() == "DIMENSION"
    reasons = ["attachment_to_measured_geometry_not_verified"]
    record = {"handle": entity.dxf.handle, "owner_block": owner,
              "insert_path": list(insert_path), "entity_type": entity.dxftype(),
              "layer": entity.dxf.layer, "dimtype": kind,
              "kind": DIMENSION_TYPES.get(kind, "unsupported") if standard_dimension else "unsupported",
              "label_status": "needs_review", "training_eligible": False,
              "dimension_associativity_verified": False,
              "definition_to_modelspace_row_vector_matrix": _matrix(transform),
              "definition_points": {}, "definition_points_modelspace": {},
              "text_override": entity.dxf.get("text", ""),
              "style_name": entity.dxf.dimstyle,
              "geometry_block": entity.dxf.get("geometry"),
              "cached_actual_measurement": _value(entity.dxf.get("actual_measurement")),
              "measurement_raw": None, "measurement_units": "DXF drawing units",
              "display_comparison": {"status": "unresolved"}}
    for name in ("defpoint", "defpoint2", "defpoint3", "defpoint4", "defpoint5"):
        if entity.dxf.hasattr(name):
            point = entity.dxf.get(name)
            record["definition_points"][name] = _value(point)
            # defpoint5 is OCS; other definition points are WCS in the owner space.
            local = entity.ocs().to_wcs(point) if name == "defpoint5" else Vec3(point)
            record["definition_points_modelspace"][name] = _value(transform.transform(local))
    roles = {0: {"defpoint2": "first_extension_origin", "defpoint3": "second_extension_origin",
                 "defpoint": "dimension_line_location"},
             1: {"defpoint2": "first_extension_origin", "defpoint3": "second_extension_origin",
                 "defpoint": "dimension_line_location"},
             3: {"defpoint": "diameter_endpoint", "defpoint4": "opposite_diameter_endpoint"},
             4: {"defpoint": "center", "defpoint4": "point_on_circle"}}
    record["definition_point_roles"] = roles.get(kind, {})
    if not standard_dimension:
        record["definition_point_roles"] = {}
        record["special_dimension_attributes"] = {
            name: _value(entity.dxf.get(name))
            for name in ("chord_point", "override_center", "jog_point", "start_angle", "end_angle",
                         "leader_point1", "leader_point2", "is_partial", "has_leader")
            if entity.dxf.hasattr(name)
        }
    record["extrusion"] = _value(entity.dxf.extrusion)
    record["angle_deg"] = _value(entity.dxf.get("angle", 0)) if entity.dxf.is_supported("angle") else None
    record["text_midpoint_ocs"] = _value(entity.dxf.get("text_midpoint"))
    if standard_dimension and kind in (2, 5):
        record["measurement_units"] = "degrees"
    elif standard_dimension and kind == 6:
        record["measurement_units"] = "DXF drawing units; ordinate vector"
    try:
        if not standard_dimension:
            reasons.append("special_dimension_measurement_unsupported")
            record["measurement_units"] = "unresolved special dimension subtype"
        else:
            record["measurement_raw"] = _value(entity.get_measurement())
            if record["measurement_raw"] is None:
                reasons.append("nonfinite_measurement")
    except (TypeError, ValueError, ArithmeticError, ezdxf.DXFError) as error:
        reasons.append("measurement_unavailable")
        record["measurement_error"] = {"type": type(error).__name__, "message": str(error)}
    if insert_path:
        reasons.append("block_instance_measurement_and_units_require_review")
    if tuple(entity.dxf.extrusion) != (0., 0., 1.):
        reasons.append("nondefault_dimension_plane")
    required = (("defpoint2", "defpoint3") if kind in (0, 1) else ("defpoint", "defpoint4") if kind in (3, 4) else ()) if standard_dimension else ()
    if any(name not in record["definition_points"] for name in required):
        reasons.append("missing_definition_points")
    try:
        style = entity.override()
        record["style_effective"] = {key: _value(style.get(key, style.dimstyle.dxf.get_default(key))) for key in STYLE_FIELDS}
        record["style_overrides"] = {key: _value(value) for key, value in style.dimstyle_attribs.items()}
    except (ValueError, ezdxf.DXFError) as error:
        record["style_effective"] = {}
        record["style_error"] = {"type": type(error).__name__, "message": str(error)}
        reasons.append("missing_or_invalid_dimension_style")
    raw_text = record["text_override"]
    mode = "suppressed" if raw_text == " " else "automatic" if raw_text in ("", "<>") else "template" if "<>" in raw_text else "explicit_override"
    record["text_mode"] = mode
    if mode != "automatic":
        reasons.append(f"dimension_text_{mode}")
    block = entity.get_geometry_block()
    texts, display_issues = _block_texts(block, document)
    record["stored_display_texts"] = texts
    reasons.extend(display_issues)
    if block is None:
        reasons.append("missing_dimension_geometry_block")
    if mode != "suppressed" and not texts:
        reasons.append("no_stored_display_text")
    if mode == "suppressed" and texts:
        reasons.append("suppressed_dimension_has_stored_text")
    effective = record["style_effective"]
    factor = effective.get("dimlfac")
    measurement = record["measurement_raw"]
    comparable = (standard_dimension and kind in (0, 1, 3, 4) and isinstance(measurement, (float, int))
                  and isinstance(factor, (float, int)) and factor > 0
                  and effective.get("dimlunit") == 2
                  and not effective.get("dimalt") and not effective.get("dimtol")
                  and not effective.get("dimlim") and "missing_definition_points" not in reasons)
    if comparable:
        expected = measurement * factor
        decimal_places = effective.get("dimdec")
        rounding = effective.get("dimrnd")
        if (math.isfinite(expected) and isinstance(decimal_places, int) and 0 <= decimal_places <= 12
                and isinstance(rounding, (float, int)) and rounding >= 0):
            tolerance = max(1e-7, 0.5 * 10**(-decimal_places) + rounding / 2)
            comparison = {"status": "unresolved", "expected_scaled_measurement": expected,
                          "absolute_tolerance": tolerance,
                          "meaning": "numeric text consistency only; not geometry attachment or validated nominal"}
            displayed = _simple_number(texts[0]["text"]) if len(texts) == 1 else None
            manual = _simple_number(raw_text) if mode == "explicit_override" else None
            comparison["stored_display_number"] = displayed
            comparison["explicit_override_number"] = manual
            checks = []
            for label, value in (("stored_display", displayed), ("explicit_override", manual)):
                if value is not None:
                    equal = math.isclose(value, expected, rel_tol=1e-8, abs_tol=tolerance)
                    comparison[f"{label}_matches_scaled_measurement"] = equal
                    checks.append(equal)
                    if not equal:
                        reasons.append(f"{label}_measurement_mismatch")
            comparison["status"] = "mismatch" if False in checks else "numeric_consistent" if checks else "unresolved"
            record["display_comparison"] = comparison
    if record["display_comparison"]["status"] == "unresolved":
        reasons.append("displayed_nominal_not_verified")
    record["reasons"] = sorted(set(reasons))
    return record


def audit_dxf_semantics(path: Path) -> dict:
    """Inspect native definitions and reachable block instances without saving DXF.

    A complete audit means inspected evidence, not accepted semantic labels.
    Unsupported measurements, unspecified units and instance transforms remain
    explicit review items. File/graph budgets bound inspection of customer input.
    """
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("DXF semantic audit requires a file of at most 50 MiB")
    source_hash = _digest(path)
    document = ezdxf.readfile(path)
    if len(document.entitydb) > MAX_ENTITIES:
        raise ValueError("DXF semantic audit entity budget exceeded")
    counts = Counter(entity.dxftype() for entity in document.modelspace())
    result = {"schema_version": "1.0", "source_sha256": source_hash,
              "status": "needs_review", "training_eligible": False,
              "source_modified": False, "units": int(document.units),
              "units_name": unit_name(document.units),
              "units_confirmed_from_external_evidence": False,
              "units_policy": "header declaration only; Unitless cannot produce physical labels",
              "modelspace_entity_counts": dict(sorted(counts.items())),
              "modelspace_dimension_count": counts["DIMENSION"],
              "layers": [], "blocks": [], "inserts": [], "dimensions": [],
              "zero_length_lines": [], "issues": [], "inventory_complete": True,
              "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_entities": MAX_ENTITIES,
                         "max_expanded_entities": MAX_EXPANDED_ENTITIES, "max_block_depth": MAX_BLOCK_DEPTH},
              "limitations": ["No source edits, regenerated dimension blocks or inferred semantic layer roles.",
                              "Point roles describe DXF definitions; attachment to measured parts is unverified.",
                              "Stored block text can be stale; numeric agreement does not establish ground truth.",
                              "Paper layouts are inventoried as blocks but not expanded into modelspace labels."]}
    if not document.units:
        result["issues"].append({"code": "unitless_physical_scale_unresolved"})
    for layer in document.layers:
        result["layers"].append({"name": layer.dxf.name, "color": layer.dxf.color,
                                 "linetype": layer.dxf.linetype, "lineweight": layer.dxf.lineweight,
                                 "off": layer.is_off(), "frozen": layer.is_frozen(),
                                 "locked": layer.is_locked(), "semantic_role": "unverified"})
    for block in document.blocks:
        result["blocks"].append({"name": block.name, "base_point": _value(block.block.dxf.base_point),
                                 "entity_counts": dict(sorted(Counter(e.dxftype() for e in block).items())),
                                 "flags": block.block.dxf.flags})
    expanded = 0

    def issue(code, entity, chain, **extra):
        result["issues"].append({"code": code, "handle": entity.dxf.handle,
                                  "insert_path": list(chain), **extra})

    def walk(space, owner, transform, chain, block_names):
        nonlocal expanded
        for entity in space:
            if expanded >= MAX_EXPANDED_ENTITIES:
                result["inventory_complete"] = False
                if not any(i["code"] == "expanded_entity_budget_exceeded" for i in result["issues"]):
                    issue("expanded_entity_budget_exceeded", entity, chain)
                return
            expanded += 1
            kind = entity.dxftype()
            if kind == "LINE" and entity.dxf.start == entity.dxf.end:
                result["zero_length_lines"].append({"handle": entity.dxf.handle, "owner_block": owner,
                                                     "insert_path": list(chain), "layer": entity.dxf.layer,
                                                     "point": _value(entity.dxf.start)})
            if kind in ("DIMENSION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION"):
                try:
                    result["dimensions"].append(_dimension(entity, document, owner, chain, transform))
                except (ValueError, TypeError, ArithmeticError, ezdxf.DXFError) as error:
                    result["dimensions"].append({
                        "handle": entity.dxf.handle, "entity_type": kind, "owner_block": owner,
                        "insert_path": list(chain), "layer": entity.dxf.layer,
                        "label_status": "needs_review", "training_eligible": False,
                        "text_mode": "unresolved", "display_comparison": {"status": "unresolved"},
                        "reasons": ["dimension_inspection_error"],
                        "error": {"type": type(error).__name__, "message": str(error)},
                    })
                    result["inventory_complete"] = False
                    issue("dimension_inspection_error", entity, chain, message=str(error))
            if kind != "INSERT":
                continue
            name = entity.dxf.name
            record = {"handle": entity.dxf.handle, "block_name": name, "owner_block": owner,
                      "insert_path": list(chain), "layer": entity.dxf.layer,
                      "insert": _value(entity.dxf.insert), "rotation_deg": entity.dxf.rotation,
                      "scale": [entity.dxf.xscale, entity.dxf.yscale, entity.dxf.zscale],
                      "rows": entity.dxf.row_count, "columns": entity.dxf.column_count,
                      "row_spacing": entity.dxf.row_spacing, "column_spacing": entity.dxf.column_spacing,
                      "attributes": [{"tag": a.dxf.tag, "text": a.dxf.text} for a in entity.attribs]}
            result["inserts"].append(record)
            if not math.isclose(abs(entity.dxf.xscale), abs(entity.dxf.yscale), rel_tol=1e-9, abs_tol=1e-12):
                issue("nonuniform_insert_xy_scale", entity, chain)
            if entity.dxf.row_count > 1 or entity.dxf.column_count > 1:
                result["inventory_complete"] = False
                issue("multiple_insert_array_not_expanded", entity, chain)
            if entity.block() is None:
                result["inventory_complete"] = False
                issue("missing_insert_block", entity, chain, block_name=name)
                continue
            if name in block_names:
                result["inventory_complete"] = False
                issue("cyclic_insert_block", entity, chain, block_name=name)
                continue
            if len(chain) >= MAX_BLOCK_DEPTH:
                result["inventory_complete"] = False
                issue("insert_depth_limit", entity, chain)
                continue
            try:
                local = entity.matrix44()
                composed = local @ transform
                record["block_to_owner_row_vector_matrix"] = _matrix(local)
                record["block_to_modelspace_row_vector_matrix"] = _matrix(composed)
                if not all(math.isfinite(v) for v in composed):
                    raise ValueError("Nonfinite INSERT transformation")
            except (ValueError, ArithmeticError, ezdxf.DXFError) as error:
                result["inventory_complete"] = False
                issue("insert_transform_error", entity, chain, message=str(error))
                continue
            walk(entity.block(), name, composed, (*chain, entity.dxf.handle), (*block_names, name))

    walk(document.modelspace(), "*Model_Space", Matrix44(), (), ())
    # ezdxf.audit may repair the in-memory document; all observations above were
    # taken first and no save operation is performed.
    audit = document.audit()
    result["audit"] = {"errors": [{"code": int(e.code), "message": e.message} for e in audit.errors],
                       "fixes": [{"code": int(e.code), "message": e.message} for e in audit.fixes],
                       "fixes_saved": False, "observations_before_in_memory_audit": True}
    if audit.errors or audit.fixes:
        result["issues"].append({"code": "dxf_audit_requires_review"})
    result["summary"] = {"expanded_entities": expanded, "dimension_instances": len(result["dimensions"]),
                         "zero_length_line_instances": len(result["zero_length_lines"]),
                         "modelspace_zero_length_lines": sum(not e["insert_path"] for e in result["zero_length_lines"]),
                         "dimension_text_modes": dict(sorted(Counter(d["text_mode"] for d in result["dimensions"]).items())),
                         "dimension_display_comparisons": dict(sorted(Counter(d["display_comparison"]["status"] for d in result["dimensions"]).items())),
                         "dimension_review_reasons": dict(sorted(Counter(r for d in result["dimensions"] for r in d["reasons"]).items()))}
    if _digest(path) != source_hash:
        raise ValueError("DXF source changed during semantic audit")
    return result
