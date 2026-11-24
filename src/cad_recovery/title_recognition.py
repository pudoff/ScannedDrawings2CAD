"""Conservative title-block proposals from OCR observations, never verified fields.

Coordinates are pixel centres in the processed image. Grid/cell segmentation is
not available here, so spatial associations are deliberately short range.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


MIN_CONFIDENCE = 0.65
_FIELDS = ("name", "document_number", "material", "authors")
_LABELS = (
    ("name", r"наименование(?:\s+изделия)?"),
    ("document_number", r"обозначение|номер\s+документа"),
    ("material", r"материал"),
    ("scale", r"масштаб"),
    ("authors", r"разраб(?:отал)?\.?|пров(?:ерил)?\.?|утв(?:ердил)?\.?|н\.?\s*контр\.?"),
)
_LABEL_PATTERNS = [(field, re.compile(r"^(?:" + pattern + r")(?=$|\s|:)\s*:?\s*(.*)$", re.I))
                   for field, pattern in _LABELS]
_HEADER = re.compile(
    r"^(?:подп(?:ись)?\.?|дата|лист(?:ов)?|лит(?:ера)?\.?|масса|изм\.?|кол\.?\s*уч\.?|"
    r"№\s*док(?:умента)?\.?|формат|фио|ф\.?\s*и\.?\s*о\.?)$", re.I)
_MATERIAL = re.compile(r"\b(?:сталь|чугун|алюминий|латунь|бронза|дюралюминий|титан|медь|сплав)(?=\W|\d|$)", re.I)
_STANDARD = re.compile(r"^(?:ГОСТ|ОСТ|ТУ|EN|DIN|ISO)\s*[-\w. /]+$", re.I)
_SURNAME = r"(?:[А-ЯЁ][а-яё]{2,}|[A-Z][a-z]{2,}|[А-ЯЁ]{3,}|[A-Z]{3,})(?:-[А-ЯЁA-Z][а-яёa-z]{2,})?"
_PERSON = re.compile(r"^(?:" + _SURNAME + r"\s+[А-ЯЁA-Z]\.\s*(?:[А-ЯЁA-Z]\.)?|"
                     + _SURNAME + r"\s+" + _SURNAME + r"\s+" + _SURNAME + r")$")
_RATIO = re.compile(r"^(?:[МM]\s*)?(\d+(?:[,.]\d+)?)\s*:\s*(\d+(?:[,.]\d+)?)$")


@dataclass(frozen=True)
class _Line:
    id: str
    raw: str
    bbox: tuple[float, float, float, float]
    confidence: float
    conflicting: bool = False

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def usable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE and not self.conflicting


def _normal(text: str) -> str:
    return " ".join(text.split())


def _label(raw: str) -> tuple[str, str, str] | None:
    text = _normal(raw)
    for field, pattern in _LABEL_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            value = match.group(1)
            label = text[:match.start(1)].rstrip(" :")
            return field, label, value
    return None


def _empty_field() -> dict:
    return {"status": "unknown", "value": None, "raw": None,
            "source_line_ids": [], "evidence": [], "alternatives": []}


def _is_value(field: str, raw: str) -> bool:
    text = _normal(raw)
    if not text or _label(text) or _HEADER.fullmatch(text):
        return False
    if field == "authors":
        words = re.findall(r"[а-яёa-z]+", text.casefold())
        blocked = {"подпись", "подп", "дата", "лист", "наименование", "разработал", "проверил", "фио"}
        return not blocked.intersection(words) and bool(_PERSON.fullmatch(text))
    if field == "material":
        return bool(_MATERIAL.search(text) or _STANDARD.fullmatch(text)
                    or re.fullmatch(r"(?=.*\d)[А-ЯЁA-Z0-9][А-ЯЁа-яёA-Za-z0-9 .-]{1,39}", text))
    if field == "document_number":
        return (bool(re.fullmatch(r"[А-ЯЁа-яёA-Za-z0-9][А-ЯЁа-яёA-Za-z0-9 ._/-]{1,79}", text))
                and any(char.isdigit() for char in text)
                and not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text)
                and not _MATERIAL.search(text) and not _STANDARD.fullmatch(text))
    if field == "name":
        return (bool(re.search(r"[А-ЯЁа-яёA-Za-z]{3,}", text))
                and not _MATERIAL.search(text) and not _STANDARD.fullmatch(text)
                and not _PERSON.fullmatch(text) and len(text) <= 120)
    if field == "scale":
        match = _RATIO.fullmatch(text)
        if not match:
            return False
        values = [float(value.replace(",", ".")) for value in match.groups()]
        return all(math.isfinite(value) and value > 0 for value in values) and math.isfinite(values[1] / values[0])
    return False


def _nearby(label: _Line, lines: list[_Line], region_width: float) -> tuple[_Line, str] | None:
    """Use the nearest visible text in a direction; never skip a wrong cell."""
    x0, y0, x1, y1 = label.bbox
    cy = (y0 + y1) / 2
    right, below = [], []
    for line in lines:
        if line.id == label.id:
            continue
        a, b, c, d = line.bbox
        height = max(label.height, line.height)
        if a >= x1 - 0.2 * height and abs((b + d) / 2 - cy) <= 0.5 * height:
            gap = max(0, a - x1)
            if gap <= min(12 * height, 0.25 * region_width):
                right.append((gap, line.id, line))
        overlap = max(0, min(c, x1) - max(a, x0))
        aligned = overlap >= 0.5 * min(c - a, x1 - x0) or abs(a - x0) <= height
        if b >= y1 - 0.2 * height and b - y1 <= 2 * height and aligned:
            below.append((max(0, b - y1), line.id, line))
    if right:
        return min(right)[2], "right"
    if below:
        return min(below)[2], "below"
    return None


def _candidate(value: str, value_lines: list[_Line], label: _Line | None,
               relationship: str, label_text: str | None = None) -> dict:
    source = ([label] if label else []) + value_lines
    return {"status": "proposed", "value": value, "raw": "\n".join(line.raw for line in value_lines),
            "source_line_ids": list(dict.fromkeys(line.id for line in source)),
            "evidence": [{"method": "explicit_label" if label else "material_marker",
                          "relationship": relationship, "label_raw": label_text,
                          "label_line_id": label.id if label else None,
                          "label_bbox_px": list(label.bbox) if label else None,
                          "value_bboxes_px": [list(line.bbox) for line in value_lines],
                          "confidence": min(line.confidence for line in source)}],
            "alternatives": []}


def _unique(candidates: list[dict]) -> list[dict]:
    result: dict[str, dict] = {}
    for candidate in candidates:
        key = _normal(candidate["value"]).casefold()
        if key not in result:
            result[key] = candidate
        else:
            prior = result[key]
            prior["source_line_ids"] = list(dict.fromkeys(prior["source_line_ids"] + candidate["source_line_ids"]))
            prior["evidence"].extend(candidate["evidence"])
    return list(result.values())


def propose_title_block(lines: list[dict], region_bbox: list[float] | None) -> dict:
    """Return reviewable proposals; absent/ambiguous evidence remains unknown.

    Input lines have id, raw, four polygon_px points and confidence in [0, 1].
    source_ref may be present; source_line_ids refer back to those OCR records.
    Only observations wholly inside the supplied title-block region are used.
    """
    report = {"status": "needs_review", "coordinate_space": "processed_px",
              "region_bbox": region_bbox, "fields": {name: _empty_field() for name in _FIELDS},
              "scale_candidates": [], "issues": []}
    if region_bbox is None:
        report["issues"].append({"code": "title_region_missing", "message": "Область штампа не найдена; поля не предполагаются."})
        return report
    if (len(region_bbox) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in region_bbox)
            or region_bbox[2] <= region_bbox[0] or region_bbox[3] <= region_bbox[1]):
        raise ValueError("region_bbox must be a finite, nonempty [x0, y0, x1, y1] rectangle")
    accepted: list[_Line] = []
    seen_ids: set[str] = set()
    for source in lines:
        try:
            polygon = source["polygon_px"]
            if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
                raise ValueError("Expected four polygon points")
            points = [(float(x), float(y)) for x, y in polygon]
            confidence = float(source["confidence"])
            if not all(math.isfinite(v) for point in points for v in point) or not 0 <= confidence <= 1:
                raise ValueError("Invalid OCR coordinates/confidence")
            identity, raw = source["id"], source["raw"]
            if not isinstance(identity, str) or not identity or identity in seen_ids or not isinstance(raw, str):
                raise ValueError("Expected unique nonempty id and raw text")
            seen_ids.add(identity)
            xs, ys = zip(*points)
            bbox = (min(xs), min(ys), max(xs), max(ys))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Empty OCR polygon")
        except (KeyError, TypeError, ValueError, OverflowError):
            report["issues"].append({"code": "title_invalid_ocr_line", "message": "Некорректная строка OCR пропущена."})
            continue
        if (bbox[0] < region_bbox[0] or bbox[1] < region_bbox[1]
                or bbox[2] > region_bbox[2] or bbox[3] > region_bbox[3]):
            continue
        if confidence < MIN_CONFIDENCE:
            report["issues"].append({"code": "title_low_confidence", "source_line_ids": [identity],
                                     "message": "Низкая уверенность OCR; строка не используется для поля штампа."})
        conflicting = bool(source.get("conflicting_line_ids"))
        if conflicting:
            report["issues"].append({"code": "title_conflicting_ocr_line", "source_line_ids": [identity],
                                     "message": "Конфликтующие чтения OCR не используются для поля штампа."})
        accepted.append(_Line(identity, raw, bbox, confidence, conflicting))

    accepted.sort(key=lambda line: (line.bbox[1], line.bbox[0], line.id))
    candidates: dict[str, list[dict]] = {name: [] for name in (*_FIELDS, "scale")}
    authors_by_role: dict[str, list[dict]] = {}
    for line in accepted:
        if not line.usable:
            continue
        label_info = _label(line.raw)
        if not label_info:
            continue
        field, label_text, inline_value = label_info
        if inline_value:
            value, value_line, relationship = inline_value, line, "inline"
        else:
            nearby = _nearby(line, accepted, region_bbox[2] - region_bbox[0])
            if nearby is None:
                continue
            value_line, relationship = nearby
            value = _normal(value_line.raw)
        if not value_line.usable or not _is_value(field, value):
            continue
        value_lines = [value_line]
        if field == "material":
            # A separate standard directly below the grade is part of the value.
            extension = _nearby(value_line, accepted, region_bbox[2] - region_bbox[0])
            if (extension and extension[1] == "below" and extension[0].usable
                    and _STANDARD.fullmatch(_normal(extension[0].raw))):
                value_lines.append(extension[0])
                value += " " + _normal(extension[0].raw)
        proposal = _candidate(value, value_lines, line, relationship, label_text)
        if field == "authors":
            role = re.sub(r"[\s.]", "", label_text.casefold())
            role = {"разработал": "разраб", "проверил": "пров", "утвердил": "утв"}.get(role, role)
            authors_by_role.setdefault(role, []).append(proposal)
        else:
            candidates[field].append(proposal)

    # Unlabelled material needs both a material word and a grade/standard. Names
    # and document numbers never use positional-only guesses.
    if not candidates["material"]:
        for line in accepted:
            if (line.usable and not _label(line.raw) and _MATERIAL.match(line.raw.strip())
                    and re.search(r"\d|ГОСТ|ОСТ|ТУ", line.raw, re.I)):
                candidates["material"].append(_candidate(_normal(line.raw), [line], None, "self"))

    for field in ("name", "document_number", "material"):
        alternatives = _unique(candidates[field])
        if len(alternatives) == 1:
            report["fields"][field] = alternatives[0]
        elif alternatives:
            report["fields"][field]["alternatives"] = alternatives
            report["issues"].append({"code": "title_field_ambiguous", "field": field,
                                     "message": "Найдены разные значения поля; требуется проверка."})
    resolved_authors = [_unique(values) for values in authors_by_role.values()]
    if resolved_authors and all(len(values) == 1 for values in resolved_authors):
        proposals = [values[0] for values in resolved_authors]
        report["fields"]["authors"] = {
            "status": "proposed", "value": "; ".join(dict.fromkeys(p["value"] for p in proposals)),
            "raw": "\n".join(p["raw"] for p in proposals),
            "source_line_ids": list(dict.fromkeys(i for p in proposals for i in p["source_line_ids"])),
            "evidence": [item for p in proposals for item in p["evidence"]], "alternatives": []}
    elif resolved_authors:
        report["fields"]["authors"]["alternatives"] = [p for values in resolved_authors for p in values]
        report["issues"].append({"code": "title_field_ambiguous", "field": "authors",
                                 "message": "Для одной роли найдены разные ФИО; требуется проверка."})
    for proposal in _unique(candidates["scale"]):
        match = _RATIO.fullmatch(proposal["value"])
        numerator, denominator = (float(v.replace(",", ".")) for v in match.groups())
        proposal.update({"paper_units": numerator, "model_units": denominator,
                         "model_per_paper": denominator / numerator})
        report["scale_candidates"].append(proposal)
    missing = [field for field in _FIELDS if report["fields"][field]["status"] == "unknown"]
    if missing:
        report["issues"].append({"code": "title_fields_incomplete", "fields": missing,
                                 "message": "Для части полей штампа недостаточно однозначных данных OCR."})
    return report
