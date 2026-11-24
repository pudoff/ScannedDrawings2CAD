"""Command line interface for the first implementation stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .pipeline import run_pipeline, write_json
from .schema import DrawingGraph


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="PDF/scan to DXF research prototype, in collaboration with 44labs")
    command.add_argument("--version", action="version", version=__version__)
    actions = command.add_subparsers(dest="command", required=True)
    audit = actions.add_parser("audit-dataset", help="Index source files, candidate pairs, duplicates and CAD defects")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    verification = actions.add_parser("verify-pairs", help="Verify candidate correspondences using drawing content")
    verification.add_argument("root", type=Path)
    verification.add_argument("--manifest", type=Path, required=True)
    verification.add_argument("--output", type=Path, required=True)
    verification.add_argument("--limit", type=int)
    splitting = actions.add_parser("split-dataset", help="Group visually verified pairs into development splits")
    splitting.add_argument("root", type=Path)
    splitting.add_argument("--manifest", type=Path, required=True)
    splitting.add_argument("--verification", type=Path, required=True)
    splitting.add_argument("--output", type=Path, required=True)
    splitting.add_argument("--seed", default="44labs-development-v1")
    evaluation = actions.add_parser("evaluate-links", help="Evaluate dimension association proposals against pixel annotations")
    evaluation.add_argument("prediction", type=Path)
    evaluation.add_argument("annotation", type=Path)
    evaluation.add_argument("--output", type=Path, required=True, help="New artifact directory")
    evaluation.add_argument("--anchor-tolerance-px", type=float, default=5)
    large = actions.add_parser("prepare-large-png", help="Prepare a bounded overview and full-resolution tiles from a large PNG")
    large.add_argument("input", type=Path)
    large.add_argument("--output", type=Path, required=True, help="New artifact directory (must not exist)")
    large.add_argument("--tile-size", type=int, default=2048)
    large.add_argument("--overlap", type=int, default=128)
    large.add_argument("--overview-max-side", type=int, default=2400)
    large.add_argument("--strip-rows", type=int, default=128)
    large.add_argument("--expected-sha256", help="Expected SHA-256 from the source intake")
    for name, help_text in (("inspect", "Read input into page graphs and source previews"),
                            ("prepare", "Prepare scan images, retain invertible transforms and candidate regions"),
                            ("detect-geometry", "Prepare scan and detect unverified pixel geometry candidates"),
                            ("recognize-text", "Prepare scan and run offline OCR, notation and title-field proposals"),
                            ("analyze-dimensions", "Prepare, detect geometry, OCR and propose dimension associations"),
                            ("convert", "Read input, export observations to DXF and report unresolved items")):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("input", type=Path)
        action.add_argument("--output", type=Path, required=True, help="New artifact directory (must not exist)")
        action.add_argument("--render-dpi", type=float, default=300 if name in {"prepare", "detect-geometry", "recognize-text", "analyze-dimensions"} else 120)
        action.add_argument("--image-dpi", type=float, help="Explicit image resolution; does not set drawing scale")
        if name in {"recognize-text", "analyze-dimensions"}:
            action.add_argument("--ocr-models", type=Path, required=True, help="Local verified PP-OCRv5 model directory; never downloaded during recognition")
        if name in {"prepare", "detect-geometry", "recognize-text", "analyze-dimensions"}:
            action.add_argument("--rotation-deg", type=int, choices=[0, 90, 180, 270], default=0, help="Clockwise rotation")
            action.add_argument("--quad", nargs=8, type=float, metavar="PIXEL", help="Single-page perspective corners: TL TR BR BL, x y in source-preview pixels")
            action.add_argument("--no-deskew", action="store_true", help="Disable automatic small-angle correction")
    export = actions.add_parser("export-graph", help="Validate a DrawingGraph JSON and export accepted geometry/dimensions")
    export.add_argument("input", type=Path)
    export.add_argument("--output", type=Path, required=True, help="New DXF filename")
    schema = actions.add_parser("schema", help="Print the DrawingGraph JSON Schema")
    schema.add_argument("--output", type=Path)
    return command


def main(argv: list[str] | None = None) -> int:
    # Windows pipes otherwise inherit a legacy codepage and fail on Unicode paths/issues.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-large-png":
            from .large_raster import prepare_large_png
            result = prepare_large_png(args.input, args.output, expected_sha256=args.expected_sha256,
                                       tile_size=args.tile_size, overlap=args.overlap,
                                       overview_max_side=args.overview_max_side, strip_rows=args.strip_rows)
            print(json.dumps({"status": result["status"], "output": str(args.output),
                              "source_size_px": result["source_size_px"], "tile_count": len(result["tiles"]),
                              "elapsed_seconds": result.get("elapsed_seconds")}, ensure_ascii=False))
            return 0 if result["status"] == "prepared" else 1
        if args.command == "evaluate-links":
            from .association_evaluation import evaluate_associations
            if args.output.exists():
                raise FileExistsError("Evaluation output already exists; choose a new directory")
            for path in (args.prediction, args.annotation):
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError("Association evaluation JSON exceeds 50 MiB")
            prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
            annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
            result = evaluate_associations(prediction, annotation, anchor_tolerance_px=args.anchor_tolerance_px)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "evaluation.json", result)
            print(json.dumps({"output": str(args.output), "evaluation": "evaluation.json"}, ensure_ascii=False))
            return 0
        if args.command in {"verify-pairs", "split-dataset"}:
            os.environ.setdefault("XDG_CACHE_HOME", str(args.output.resolve().parent / ".cache"))
            if args.command == "verify-pairs":
                from .pairing import verify_pairs
                result = verify_pairs(args.root, args.manifest, args.output, limit=args.limit)
            else:
                from .splits import write_splits
                result = write_splits(args.root, args.manifest, args.verification, args.output, seed=args.seed)
            print(json.dumps({"output": str(args.output), "summary": result.get("summary", {})}, ensure_ascii=False))
            return 0
        if args.command == "schema":
            value = DrawingGraph.model_json_schema()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, value)
            else:
                print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "audit-dataset":
            os.environ.setdefault("XDG_CACHE_HOME", str(args.output.resolve() / ".cache"))
            from .dataset import write_manifest
            # Validate root before reserving an output directory.
            if not args.root.is_dir():
                raise ValueError(f"Dataset directory does not exist: {args.root}")
            if args.output.resolve().is_relative_to(args.root.resolve()):
                raise ValueError("Audit output must be outside the source dataset")
            args.output.mkdir(parents=True, exist_ok=False)
            result = write_manifest(args.root, args.output)
            print(json.dumps({"output": str(args.output), "summary": result.get("summary", {})}, ensure_ascii=False))
            return 0
        if args.command == "export-graph":
            os.environ.setdefault("XDG_CACHE_HOME", str(args.output.resolve().parent / ".cache"))
            from .exporter import export_dxf
            report_path = args.output.with_suffix(".report.json")
            if args.output.exists() or report_path.exists():
                raise FileExistsError("DXF or report output already exists; choose a new filename")
            graph = DrawingGraph.model_validate_json(args.input.read_text(encoding="utf-8"))
            result = export_dxf(graph, args.output)
            write_json(report_path, result)
        else:
            quad = getattr(args, "quad", None)
            result = run_pipeline(args.input, args.output, export=args.command == "convert",
                                  render_dpi=args.render_dpi, image_dpi=args.image_dpi,
                                  prepare=args.command in {"prepare", "detect-geometry", "recognize-text"}, rotation_deg=getattr(args, "rotation_deg", 0),
                                  perspective_quad=[quad[n:n + 2] for n in range(0, 8, 2)] if quad else None,
                                  auto_deskew=not getattr(args, "no_deskew", False), detect=args.command == "detect-geometry",
                                  ocr=args.command == "recognize-text", ocr_models=getattr(args, "ocr_models", None),
                                  associate=args.command == "analyze-dimensions")
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "elapsed_seconds": result.get("elapsed_seconds")}, ensure_ascii=False))
        return 1 if result["status"] == "failed" else 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
