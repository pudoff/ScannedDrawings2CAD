"""Read-only dataset inventory, conservative pairing, and format audits.

Name matches are candidates, not verified ground truth. Exact hashes establish
byte identity only; neither mechanism assigns semantic families or data splits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

import ezdxf
from ezdxf.units import unit_name
from PIL import Image
from pypdf import PdfReader


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SCHEMA_VERSION = "1.0"


@contextmanager
def _capture_warnings(library: str, item: dict[str, Any], path: Path):
    """Retain bounded diagnostic samples instead of flooding batch output."""
    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            item["warning_count"] += 1
            if len(item["warning_samples"]) < 10:
                item["warning_samples"].append({
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage().replace(str(path), path.name),
                })

    logger = logging.getLogger(library)
    handler = Collector(level=logging.WARNING)
    original_level, original_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def normalize_stem(stem: str) -> str:
    """Normalize Unicode, case, and whitespace, preserving meaningful suffixes."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", stem)).strip().casefold()


def _kind(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return {".pdf": "pdf", ".dxf": "dxf"}.get(suffix, "other")


def _error(stage: str, error: Exception, path: Path) -> dict[str, str]:
    # Keep reports portable even when a library includes the absolute filename.
    message = str(error).replace(str(path), path.name)
    return {"stage": stage, "type": type(error).__name__, "message": message}


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric metadata")
    return result


def _audit_pdf(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    reader = PdfReader(path, strict=False)
    metadata: dict[str, Any] = {"encrypted": reader.is_encrypted, "pages": []}
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF requires a password")
    metadata["page_count"] = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        item: dict[str, Any] = {"number": index}
        try:
            user_unit = _finite(page.get("/UserUnit", 1))
            width_pt = _finite(page.mediabox.width) * user_unit
            height_pt = _finite(page.mediabox.height) * user_unit
            item.update(
                width_pt=width_pt,
                height_pt=height_pt,
                width_mm=round(width_pt * 25.4 / 72, 6),
                height_mm=round(height_pt * 25.4 / 72, 6),
                rotation=int(page.get("/Rotate", 0)),
                user_unit=user_unit,
            )
        except Exception as error:
            errors.append(_error(f"pdf.page[{index}].size", error, path))
        try:
            text = page.extract_text() or ""
            item["text_char_count"] = len(text)
            item["text_sample"] = text[:240]
        except Exception as error:
            errors.append(_error(f"pdf.page[{index}].text", error, path))
        try:
            # Counts distinct embedded image resources, not placements or pixels.
            item["image_count"] = len(page.images)
        except Exception as error:
            errors.append(_error(f"pdf.page[{index}].images", error, path))
        metadata["pages"].append(item)
    return metadata


def _audit_entry(entry: Any) -> dict[str, Any]:
    entity = entry.entity
    return {
        "code": int(entry.code),
        "message": entry.message,
        "entity_type": entity.dxftype() if entity is not None else None,
        "entity_handle": entity.dxf.get("handle") if entity is not None else None,
    }


def _audit_dxf(path: Path) -> dict[str, Any]:
    document = ezdxf.readfile(path)
    space = document.modelspace()
    counts = Counter(entity.dxftype() for entity in space)
    # Measure the input before ezdxf.audit() applies in-memory repairs.
    zero_length_lines = sum(
        entity.dxf.start == entity.dxf.end for entity in space.query("LINE")
    )
    metadata: dict[str, Any] = {
        "version": document.dxfversion,
        "encoding": document.encoding,
        "units": int(document.units),
        "units_name": unit_name(document.units),
        "modelspace_entity_count": sum(counts.values()),
        "modelspace_entities": dict(sorted(counts.items())),
        "dimension_count": counts["DIMENSION"],
        "zero_length_line_count": zero_length_lines,
    }
    audit = document.audit()
    metadata["audit"] = {
        "error_count": len(audit.errors),
        "fix_count": len(audit.fixes),
        "errors": [_audit_entry(entry) for entry in audit.errors],
        "fixes": [_audit_entry(entry) for entry in audit.fixes],
        "fixes_saved": False,
    }
    return metadata


def _audit_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as picture:
        dpi = picture.info.get("dpi")
        if isinstance(dpi, (int, float)):
            dpi = (dpi, dpi)
        metadata = {
            "format": picture.format,
            "width_px": picture.width,
            "height_px": picture.height,
            "mode": picture.mode,
            "frame_count": getattr(picture, "n_frames", 1),
            "dpi": [_finite(value) for value in dpi] if dpi else None,
        }
        # Decode every frame to detect truncated payloads, not only bad headers.
        for frame in range(metadata["frame_count"]):
            picture.seek(frame)
            picture.load()
    return metadata


def _inspect_file(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    kind = _kind(path)
    ancillary = relative.as_posix().casefold() == "архитектура.pdf"
    item: dict[str, Any] = {
        "path": relative.as_posix(),
        "set": relative.parts[0] if len(relative.parts) > 1 else ".",
        "kind": kind,
        "role": "ancillary" if ancillary else "drawing" if kind != "other" else "other",
        "normalized_stem": normalize_stem(path.stem),
        "size_bytes": None,
        "sha256": None,
        "status": "not_audited" if kind == "other" else "ok",
        "metadata": {},
        "errors": [],
        "warning_count": 0,
        "warning_samples": [],
    }
    if ancillary:
        item["role_reason"] = "Repository architecture document, not a drawing sample"
    if path.is_symlink():
        item["status"] = "error"
        item["errors"].append({"stage": "read", "type": "SymlinkSkipped", "message": "Symbolic links are not followed"})
        return item
    try:
        item["size_bytes"] = path.stat().st_size
        with path.open("rb") as stream:
            item["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    except Exception as error:
        item["errors"].append(_error("read", error, path))
    try:
        library = {"pdf": "pypdf", "dxf": "ezdxf", "image": "PIL"}.get(kind, "cad_recovery.dataset")
        with _capture_warnings(library, item, path):
            if kind == "pdf":
                item["metadata"] = _audit_pdf(path, item["errors"])
            elif kind == "dxf":
                item["metadata"] = _audit_dxf(path)
            elif kind == "image":
                item["metadata"] = _audit_image(path)
    except Exception as error:
        item["errors"].append(_error(kind, error, path))
    if item["errors"]:
        item["status"] = "partial" if item["metadata"] else "error"
    elif item["warning_count"]:
        item["status"] = "ok_with_warnings"
    return item


def _pair_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in files:
        if item["role"] == "drawing":
            groups[item["set"], item["normalized_stem"]][item["kind"]].append(item["path"])
    candidates: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    for (set_name, stem), kinds in sorted(groups.items()):
        targets = kinds.get("dxf", [])
        sources_present = False
        for input_kind in ("pdf", "image"):
            sources = kinds.get(input_kind, [])
            sources_present = sources_present or bool(sources)
            if sources and targets:
                if len(sources) == len(targets) == 1:
                    candidates.append({
                        "set": set_name,
                        "normalized_stem": stem,
                        "input_kind": input_kind,
                        "input_path": sources[0],
                        "target_path": targets[0],
                        "evidence": "unique_normalized_stem_within_set",
                        "verified": False,
                    })
                else:
                    ambiguous.append({
                        "set": set_name,
                        "normalized_stem": stem,
                        "input_kind": input_kind,
                        "input_paths": sources,
                        "target_paths": targets,
                        "reason": "non_unique_normalized_stem",
                    })
            elif sources:
                unmatched.extend({"path": path, "reason": "no_dxf_with_same_normalized_stem_in_set"} for path in sources)
        if targets and not sources_present:
            unmatched.extend({"path": path, "reason": "no_pdf_or_image_with_same_normalized_stem_in_set"} for path in targets)
    return {
        "pair_candidates": candidates,
        "ambiguous_pairs": ambiguous,
        "unmatched_files": sorted(unmatched, key=lambda item: item["path"]),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    """Audit all files under root without modifying them or creating splits.

    The first relative directory identifies a dataset. PDF and image candidates
    are matched to DXF independently, using only unique normalized file stems.
    A parse failure does not remove a file from the inventory or name matching.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted((path for path in root.rglob("*") if path.is_file() or path.is_symlink()), key=lambda path: path.relative_to(root).as_posix())
    files = [_inspect_file(path, root) for path in paths]
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in files:
        if item["sha256"]:
            hashes[item["sha256"]].append(item)
    duplicates = [
        {"sha256": digest, "size_bytes": items[0]["size_bytes"], "paths": [item["path"] for item in items]}
        for digest, items in sorted(hashes.items()) if len(items) > 1
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "root": ".",
        "policy": {
            "set_scope": "first_relative_directory_or_dot",
            "stem_normalization": "NFKC, collapsed whitespace, stripped whitespace, casefold; punctuation and suffixes preserved",
            "pairing": "PDF and image to DXF separately; unique normalized stem within set; unverified candidates only",
            "duplicate_meaning": "Exact byte identity only; no semantic family or split is inferred",
            "pdf_image_count": "Distinct embedded image resources per page, not placements",
            "dxf_counts": "Modelspace before in-memory audit fixes; block contents are not expanded",
            "source_files_modified": False,
        },
        "files": files,
        "exact_duplicate_groups": duplicates,
        **_pair_files(files),
    }
    manifest["summary"] = _summarize(manifest)
    return manifest


def _summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest["files"]
    pdfs = [item for item in files if item["kind"] == "pdf"]
    dxfs = [item for item in files if item["kind"] == "dxf"]
    pages = [page for item in pdfs for page in item["metadata"].get("pages", [])]
    sets = {}
    for set_name in sorted({item["set"] for item in files}):
        selected = [item for item in files if item["set"] == set_name]
        sets[set_name] = {
            "file_count": len(selected),
            "by_kind": dict(sorted(Counter(item["kind"] for item in selected).items())),
            "pair_candidates": sum(pair["set"] == set_name for pair in manifest["pair_candidates"]),
        }
    return {
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] or 0 for item in files),
        "by_kind": dict(sorted(Counter(item["kind"] for item in files).items())),
        "by_role": dict(sorted(Counter(item["role"] for item in files).items())),
        "by_status": dict(sorted(Counter(item["status"] for item in files).items())),
        "files_with_warnings": sum(item["warning_count"] > 0 for item in files),
        "warning_count": sum(item["warning_count"] for item in files),
        "sets": sets,
        "pair_candidate_count": len(manifest["pair_candidates"]),
        "ambiguous_pair_group_count": len(manifest["ambiguous_pairs"]),
        "unmatched_file_count": len(manifest["unmatched_files"]),
        "exact_duplicate_group_count": len(manifest["exact_duplicate_groups"]),
        "exact_duplicate_file_count": sum(len(group["paths"]) for group in manifest["exact_duplicate_groups"]),
        "redundant_bytes": sum((len(group["paths"]) - 1) * group["size_bytes"] for group in manifest["exact_duplicate_groups"]),
        "pdf": {
            "page_count": len(pages),
            "pages_with_text": sum(page.get("text_char_count", 0) > 0 for page in pages),
            "pages_with_images": sum(page.get("image_count", 0) > 0 for page in pages),
        },
        "dxf": {
            "versions": dict(sorted(Counter(item["metadata"].get("version", "unknown") for item in dxfs).items())),
            "units": dict(sorted(Counter(item["metadata"].get("units_name", "unknown") for item in dxfs).items())),
            "files_with_dimensions": sum(item["metadata"].get("dimension_count", 0) > 0 for item in dxfs),
            "dimension_count": sum(item["metadata"].get("dimension_count", 0) for item in dxfs),
            "files_with_zero_length_lines": sum(item["metadata"].get("zero_length_line_count", 0) > 0 for item in dxfs),
            "zero_length_line_count": sum(item["metadata"].get("zero_length_line_count", 0) for item in dxfs),
            "audit_error_count": sum(item["metadata"].get("audit", {}).get("error_count", 0) for item in dxfs),
            "audit_fix_count": sum(item["metadata"].get("audit", {}).get("fix_count", 0) for item in dxfs),
        },
    }


def _markdown(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Аудит датасета", "",
        f"Всего файлов: **{summary['file_count']}**, объём: **{summary['total_bytes']:,} байт**.",
        "Пути указаны относительно корня датасета. Исходные файлы не изменены.", "",
        "| Набор | PDF | DXF | Изображения | Прочие | Кандидаты пар |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, subset in summary["sets"].items():
        counts = subset["by_kind"]
        lines.append(f"| {name} | {counts.get('pdf', 0)} | {counts.get('dxf', 0)} | {counts.get('image', 0)} | {counts.get('other', 0)} | {subset['pair_candidates']} |")
    lines.extend([
        "", "## Соответствия и дубликаты", "",
        f"- Кандидаты пар: {summary['pair_candidate_count']}; требуют подтверждения содержания.",
        f"- Неоднозначные группы: {summary['ambiguous_pair_group_count']}.",
        f"- Файлы без соответствия: {summary['unmatched_file_count']}.",
        f"- Точные дубли: {summary['exact_duplicate_group_count']} групп, {summary['exact_duplicate_file_count']} файлов, {summary['redundant_bytes']:,} повторных байт.",
        "- SHA-256 подтверждает только побайтовое совпадение; семантическое родство деталей не установлено.",
        "- Train/validation/test автоматически не создаются.",
        "- Сопоставление: NFKC, регистр и пробелы; суффиксы/пунктуация сохранены. Уникальность проверяется внутри первого каталога отдельно для PDF и изображений.",
        "- `Архитектура.pdf` в корне явно считается сопроводительным документом и исключается из пар.",
        "", "## PDF и DXF", "",
        f"- PDF: {summary['pdf']['page_count']} прочитанных страниц (включая сопроводительные документы), {summary['pdf']['pages_with_text']} с текстом, {summary['pdf']['pages_with_images']} с изображениями.",
        f"- Предупреждения библиотек: {summary['warning_count']} в {summary['files_with_warnings']} файлах; первые 10 на файл сохранены в manifest.json. Такие файлы требуют проверки извлечённого текста и структуры.",
        "- Количество изображений — уникальные встроенные ресурсы страницы; наличие текста/изображений не определяет пригодность для восстановления.",
        f"- DXF с DIMENSION в modelspace: {summary['dxf']['files_with_dimensions']}; объектов DIMENSION: {summary['dxf']['dimension_count']}.",
        f"- Нулевые LINE: {summary['dxf']['zero_length_line_count']} в {summary['dxf']['files_with_zero_length_lines']} DXF.",
        f"- Аудит ezdxf: ошибок {summary['dxf']['audit_error_count']}, исправлений в памяти {summary['dxf']['audit_fix_count']}; исправления не сохранены в исходниках.",
        "- Объекты внутри BLOCK/INSERT не разворачиваются; счётчики DXF относятся к modelspace до аудита.",
        f"- Версии DXF: {json.dumps(summary['dxf']['versions'], ensure_ascii=False, sort_keys=True)}.",
        f"- Единицы DXF: {json.dumps(summary['dxf']['units'], ensure_ascii=False, sort_keys=True)}.",
        "", "## Неподтверждённые соответствия", "",
    ])
    if not manifest["unmatched_files"] and not manifest["ambiguous_pairs"]:
        lines.append("Нет файлов без кандидата или с неоднозначным именем.")
    for item in manifest["unmatched_files"]:
        lines.append(f"- `{item['path']}`: `{item['reason']}`.")
    for group in manifest["ambiguous_pairs"]:
        paths = ", ".join(f"`{path}`" for path in group["input_paths"] + group["target_paths"])
        lines.append(f"- Неоднозначно ({group['input_kind']} → DXF): {paths}.")
    lines.extend(["", "## Ошибки чтения и разбора", ""])
    failed = [item for item in manifest["files"] if item["errors"]]
    if not failed:
        lines.append("Ошибок чтения и разбора не обнаружено.")
    for item in failed:
        for error in item["errors"]:
            message = error["message"].replace("\n", " ")
            lines.append(f"- `{item['path']}` [{error['stage']}]: {error['type']}: {message}")
    lines.extend(["", "Полные метаданные каждого файла, группы SHA-256 и кандидаты пар: `manifest.json`.", ""])
    return "\n".join(lines)


def write_manifest(root: Path, output: Path) -> dict[str, Any]:
    """Write deterministic manifest.json and summary.md outside the dataset."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Audit output must be outside the dataset root to avoid indexing its own results")
    manifest = build_manifest(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output / "summary.md").write_text(_markdown(manifest), encoding="utf-8", newline="\n")
    return manifest
