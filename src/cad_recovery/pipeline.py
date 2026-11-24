"""One local, reproducible input-to-artifacts route; no reference DXF is read."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .intake import read_input
from .schema import Issue


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_pipeline(source: Path, output: Path, *, export: bool = True,
                 render_dpi: float = 120, image_dpi: float | None = None,
                 prepare: bool = False, rotation_deg: int = 0,
                 perspective_quad: list | None = None, auto_deskew: bool = True,
                 detect: bool = False, ocr: bool = False, ocr_models: Path | None = None,
                 associate: bool = False) -> dict:
    started = time.perf_counter()
    detect = detect or associate
    ocr = ocr or associate
    prepare = prepare or detect or ocr
    output = Path(output)
    # Reprocessing creates a new artifact directory instead of replacing an earlier result.
    output.mkdir(parents=True, exist_ok=False)
    try:
        engine = None
        if ocr:
            if ocr_models is None:
                raise ValueError("OCR requires an explicit local model directory (--ocr-models)")
            from .ocr_engine import LocalOCREngine
            engine = LocalOCREngine(ocr_models)
        graphs = read_input(source, output, render_dpi=render_dpi, image_dpi=image_dpi)
        if prepare and perspective_quad is not None and len(graphs) != 1:
            raise ValueError("A perspective quadrilateral applies to a single page; process pages separately")
        reports = []
        for i, graph in enumerate(graphs, 1):
            prefix = f"page-{i:03d}"
            preprocessing = None
            if prepare:
                from .preprocessing import prepare_scan
                preview = output / f"{prefix}-source.png"
                preprocessing = prepare_scan(preview, output, prefix=prefix, rotation_deg=rotation_deg,
                                             perspective_quad=perspective_quad, auto_deskew=auto_deskew)
                with Image.open(preview) as image:
                    width, height = image.size
                sx, sy = graph.page.width_mm / width, graph.page.height_mm / height
                # Pixel center convention: first pixel is (0,0), page edges at -0.5.
                input_to_paper = np.array([[sx, 0, sx / 2], [0, -sy, graph.page.height_mm - sy / 2], [0, 0, 1]])
                inverse = np.array(preprocessing["transforms"]["processed_to_input"])
                processed_to_paper = input_to_paper @ inverse
                mapping = {"input_pixel_to_paper_mm": input_to_paper.tolist(),
                           "processed_pixel_to_paper_mm": processed_to_paper.tolist(),
                           "paper_mm_to_processed_pixel": np.linalg.inv(processed_to_paper).tolist(),
                           "pixel_convention": "centers, x right, y down; page boundaries at -0.5",
                           "physical_scale_confirmed": False}
                preprocessing["page_mapping"] = mapping
                write_json(output / preprocessing["artifacts"]["report"], preprocessing)
                graph.provenance["preprocessing"] = {"report": preprocessing["artifacts"]["report"],
                                                     "transforms": preprocessing["transforms"], "page_mapping": mapping}
                graph.issues.append(Issue(code="preprocessing_only", message="Растр подготовлен; точная геометрия и размерные связи ещё не восстановлены."))
            detection = None
            if detect:
                from .geometry_detection import detect_geometry
                detection = detect_geometry(output / preprocessing["artifacts"]["grayscale"], output,
                                            prefix=prefix, pixel_to_paper_mm=preprocessing["page_mapping"]["processed_pixel_to_paper_mm"])
                graph.provenance["geometry_detection"] = {
                    "report": detection["artifacts"]["report"],
                    "candidate_count": len(detection["candidates"]),
                    "coordinate_space": "processed_pixel_centers",
                    "acceptance": "proposed_only; no primitive or dimension promoted",
                    "search_profile": detection['analysis'].get('search_profile', 'bounded_no_hough_proposals'),
                    "search_incomplete": any(issue['code'] in ('geometry_candidate_limit', 'skeleton_iteration_limit')
                                             for issue in detection['issues']),
                    "search_scope": "bounded pixel proposals; no exhaustive geometric recovery claim",
                    "search_counters": {key: detection['analysis'].get(key, 0) for key in
                        ('hough_circle_proposals', 'cheap_screen_examined', 'validation_queue_retained',
                         'validation_queue_dropped', 'circle_validation_examined', 'hough_circles_added',
                         'regional_short_line_proposals', 'regional_short_lines_retained')},
                }
                if graph.provenance['geometry_detection']['search_incomplete']:
                    graph.issues.append(Issue(code='geometry_search_incomplete',
                        message='Достигнут бюджет геометрического поиска; отсутствие кандидата не означает отсутствие элемента.'))
                graph.issues.append(Issue(code="geometry_candidates_unverified", message="Найдены кандидаты геометрии в пикселях; классификация и точность не подтверждены."))
            recognition = None
            if ocr:
                from .ocr import recognize_text
                title_region = next((region["bbox_px"] for region in preprocessing["regions"]
                                     if region["id"] == "title-block-candidate"), None)
                recognition = recognize_text(
                    output / preprocessing["artifacts"]["grayscale"], output, engine=engine, prefix=prefix,
                    pixel_to_paper_mm=preprocessing["page_mapping"]["processed_pixel_to_paper_mm"],
                    processed_to_input=preprocessing["transforms"]["processed_to_input"], title_bbox=title_region)
                graph.provenance["ocr"] = {
                    "report": recognition["artifacts"]["report"], "line_count": len(recognition["lines"]),
                    "notation_count": len(recognition["notations"]), "coverage": recognition["coverage"],
                    "acceptance": "observations and proposals only; no text, title field, dimension or scale promoted",
                }
                graph.issues.append(Issue(code="ocr_candidates_unverified", message="Сохранены OCR-наблюдения, варианты надписей и кандидаты полей штампа; требуется проверка и связь с геометрией."))
            arrows = None
            associations = None
            if associate:
                from .arrow_detection import detect_arrows
                from .dimension_association import associate_dimensions
                prepared_image = output / preprocessing["artifacts"]["grayscale"]
                arrows = detect_arrows(prepared_image, output, prefix=prefix, geometry=detection,
                                       text_polygons=[line["polygon_px"] for line in recognition["lines"]])
                associations = associate_dimensions(prepared_image, output, prefix=prefix,
                                                     geometry=detection, ocr=recognition, arrows=arrows)
                graph.provenance["dimension_associations"] = {
                    "report": associations["artifacts"]["report"], "summary": associations["summary"],
                    "arrow_report": arrows["artifacts"]["report"],
                    "acceptance": "proposals only; geometry, dimensions and scale remain unverified",
                }
                graph.issues.append(Issue(code="dimension_associations_unverified", message="Сохранены предложения размерных связей и причины отказов; масштаб и геометрическая точность ещё не подтверждены."))
            (output / f"{prefix}-graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")
            if export:
                os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
                from .exporter import export_dxf
                report = export_dxf(graph, output / f"{prefix}.dxf")
            else:
                issues = graph.review_issues()
                report = {"status": "needs_review" if issues else "succeeded",
                          "issues": [issue.model_dump() for issue in issues]}
            report["page_index"] = i - 1
            report["source_kind"] = graph.page.kind
            report["artifacts"] = {"graph": f"{prefix}-graph.json", "source_preview": f"{prefix}-source.png"}
            if preprocessing is not None:
                report["preprocessing"] = preprocessing
                report["artifacts"].update(preprocessing["artifacts"])
            if detection is not None:
                report["geometry_detection"] = detection
                report["artifacts"]["geometry_candidates"] = detection["artifacts"]["report"]
                report["artifacts"]["geometry_preview"] = detection["artifacts"]["preview"]
            if recognition is not None:
                report["ocr"] = recognition
                report["artifacts"]["ocr_report"] = recognition["artifacts"]["report"]
                report["artifacts"]["ocr_preview"] = recognition["artifacts"]["preview"]
            if associations is not None:
                report["arrows"] = arrows
                report["dimension_associations"] = associations
                report["artifacts"]["arrow_report"] = arrows["artifacts"]["report"]
                report["artifacts"]["arrow_preview"] = arrows["artifacts"]["preview"]
                report["artifacts"]["dimension_associations"] = associations["artifacts"]["report"]
                report["artifacts"]["dimension_associations_preview"] = associations["artifacts"]["preview"]
            if export and (output / f"{prefix}.dxf").exists():
                report["artifacts"]["dxf"] = f"{prefix}.dxf"
            write_json(output / f"{prefix}-report.json", report)
            reports.append(report)
        status = ("failed" if any(r["status"] == "failed" for r in reports) else
                  "needs_review" if any(r["status"] == "needs_review" for r in reports) else "succeeded")
        result = {
            "status": status, "source_sha256": graphs[0].source.sha256,
            "page_count": len(graphs), "elapsed_seconds": round(time.perf_counter() - started, 4),
            "code_version": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("pypdf", "pypdfium2", "ezdxf", "pydantic", "pillow", "numpy", "opencv-python", "PyMuPDF", "rapidocr", "onnxruntime")},
            "parameters": {"render_dpi": render_dpi, "image_dpi": image_dpi, "export_dxf": export,
                           "prepare": prepare, "rotation_deg": rotation_deg, "perspective_quad": perspective_quad,
                           "auto_deskew": auto_deskew, "detect_geometry": detect, "ocr": ocr,
                           "ocr_models": str(ocr_models) if ocr_models is not None else None,
                           "associate_dimensions": associate},
            "pages": reports,
            "limitations": ["Наблюдения и предложения связей требуют проверки; масштаб и точная геометрия ещё не восстановлены.",
                            "Сверка с эталоном и открытие в CAD не выполнялись."],
        }
        write_json(output / "report.json", result)
        return result
    except Exception as exc:
        write_json(output / "report.json", {"status": "failed", "error": str(exc),
                                          "elapsed_seconds": round(time.perf_counter() - started, 4)})
        raise
