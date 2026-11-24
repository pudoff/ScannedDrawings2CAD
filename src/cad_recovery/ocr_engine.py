"""Pinned local OCR backend. Model provisioning is a separate explicit command."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np

DEFAULT_MODEL_MANIFEST = Path(__file__).resolve().parent / "data" / "ocr-models-v1.json"
MAX_PIXELS = 25_000_000
_BACKEND_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _preflight(model_dir: Path) -> tuple[dict, dict, list[dict]]:
    manifest = json.loads(DEFAULT_MODEL_MANIFEST.read_text(encoding="utf-8"))
    models = manifest["models"]
    if len(models) != 3 or {item["role"] for item in models} != {"det", "rec", "cls"}:
        raise ValueError("OCR manifest must contain exactly det, rec and cls models")
    paths, evidence = {}, []
    for item in models:
        filename, digest = item["filename"], item["sha256"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.onnx", filename) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid OCR model filename or SHA-256 in pinned manifest")
        path = model_dir / filename
        if path.is_symlink() or not path.resolve().is_relative_to(model_dir):
            raise ValueError("OCR models must be regular local files inside model_dir")
        if not path.is_file():
            raise FileNotFoundError(
                f"Local OCR model missing: {path}. Run scripts/fetch_ocr_models.py explicitly during setup; "
                "OCR never downloads models.")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(f"Local OCR model SHA-256 mismatch: {path.name}; file was not modified")
        paths[item["role"]] = path
        evidence.append({"role": item["role"], "filename": filename, "sha256": actual,
                         "bytes": path.stat().st_size})
    if len({path.name for path in paths.values()}) != 3:
        raise ValueError("OCR model filenames must be distinct")
    return manifest, paths, evidence


@contextmanager
def _offline_backend_scope():
    """Deny RapidOCR's hidden model/dictionary/font download paths.

    Its downloader is shared by all three components. Hold a process-local lock
    while replacing its descriptor and while using the mutable OCR instance;
    restore it even after an exception. The setup script uses urllib separately.
    """
    with _BACKEND_LOCK:
        from rapidocr.utils.download_file import DownloadFile

        previous = vars(DownloadFile)["run"]

        def reject_download(*args, **kwargs):
            raise RuntimeError("Automatic RapidOCR downloads are disabled; provision local models explicitly")

        DownloadFile.run = staticmethod(reject_download)
        try:
            yield
        finally:
            DownloadFile.run = previous


def _initialize_rapidocr(paths: dict, cpu_threads: int):
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    params = {
        "Global.text_score": 0.0,
        "Global.use_det": True,
        "Global.use_cls": True,
        "Global.use_rec": True,
        "Global.log_level": "warning",
        "Global.max_side_len": 2000,
        "Global.return_word_box": False,
        "Global.return_single_char_box": False,
        "Global.model_root_dir": str(paths["det"].parent),
        "Det.limit_type": "max",
        "Det.limit_side_len": 2000,
        "Det.lang_type": LangDet.CH,
        "Cls.lang_type": LangCls.CH,
        "Rec.lang_type": LangRec.ESLAV,
        "Rec.rec_keys_path": None,
        "Rec.rec_batch_num": 6,
        "EngineConfig.onnxruntime.intra_op_num_threads": cpu_threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
    }
    for role, path in paths.items():
        section = role.capitalize()
        params.update({f"{section}.model_path": str(path),
                       f"{section}.engine_type": EngineType.ONNXRUNTIME,
                       f"{section}.model_type": ModelType.MOBILE,
                       f"{section}.ocr_version": OCRVersion.PPOCRV5})
    backend = RapidOCR(params=params)
    if not backend.text_rec.session.have_key("character"):
        raise ValueError("Pinned OCR recognizer must contain its character dictionary in ONNX metadata")
    return backend


class LocalOCREngine:
    """Initialize once, then recognize multiple BGR uint8 pages/tiles locally.

    Results keep backend strings and confidence values, without score filtering
    or normalization. Polygons use the coordinates of the supplied image.
    Confidence is an OCR score, not a calibrated correctness probability.
    """

    def __init__(self, model_dir: Path):
        root = Path(model_dir).resolve()
        manifest, paths, models = _preflight(root)
        versions = {name: importlib.metadata.version(name) for name in ("rapidocr", "onnxruntime")}
        for name, actual in versions.items():
            expected = manifest[f"{name}_version"]
            if actual != expected:
                raise RuntimeError(f"OCR backend requires {name}=={expected}; found {actual}")
        cpu_threads = min(4, os.cpu_count() or 1)
        with _offline_backend_scope():
            self._backend = _initialize_rapidocr(paths, cpu_threads)
        self._provenance = {
            "engine": "RapidOCR", "engine_version": versions["rapidocr"],
            "onnxruntime_version": versions["onnxruntime"],
            "provider": "CPUExecutionProvider", "intra_op_threads": cpu_threads, "inter_op_threads": 1,
            "model_manifest_id": manifest["id"], "model_manifest_sha256": _sha256(DEFAULT_MODEL_MANIFEST),
            "model_source_manifest": manifest["source_manifest"], "models": models,
            "offline": True, "automatic_downloads": "blocked", "recognition_score_threshold": 0.0,
            "recognizer_language": "eslav", "character_dictionary": "embedded_onnx_metadata",
            "input_format": "uint8 BGR", "polygon_coordinate_space": "input_image_pixels",
            "max_input_pixels": MAX_PIXELS, "backend_max_side_len": 2000,
            "detection_box_threshold": 0.5,
            "limitations": ["Confidence scores are uncalibrated.",
                            "The backend drops empty decoded strings; detector thresholds still apply.",
                            "No engineering symbol correction or dimension association is inferred."],
        }

    @property
    def provenance(self) -> dict:
        return deepcopy(self._provenance)

    def recognize(self, image: np.ndarray) -> list[dict]:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError("OCR image must be a nonempty uint8 BGR array with shape (height, width, 3)")
        if image.shape[0] * image.shape[1] > MAX_PIXELS:
            raise ValueError("OCR image exceeds 25 megapixel limit")
        # Protect callers from upstream preprocessing implementations that mutate
        # their input. Negative-stride crops are also made contiguous here.
        local = np.array(image, copy=True, order="C")
        with _offline_backend_scope():
            output = self._backend(local, text_score=0.0, use_det=True, use_cls=True, use_rec=True)
        boxes, texts, scores = (getattr(output, key, None) for key in ("boxes", "txts", "scores"))
        if all(value is None for value in (boxes, texts, scores)):
            return []
        if any(value is None for value in (boxes, texts, scores)) or not len(boxes) == len(texts) == len(scores):
            raise RuntimeError("OCR backend returned incomplete or inconsistent text/box/confidence arrays")
        result = []
        for polygon, raw, confidence in zip(boxes, texts, scores, strict=True):
            polygon = np.asarray(polygon, dtype=float)
            confidence = float(confidence)
            if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
                raise RuntimeError("OCR backend returned an invalid quadrilateral")
            if not isinstance(raw, str) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise RuntimeError("OCR backend returned invalid text or confidence")
            result.append({"raw": raw, "confidence": confidence, "polygon_px": polygon.tolist()})
        return result
