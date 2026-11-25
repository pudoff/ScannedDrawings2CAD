"""Validate current guarded payload, not the unused historical Web shell."""
import importlib,os
from pathlib import Path
from .compact_runtime_v1 import ROOT,verify_environment


def preflight():
    if os.name!='nt':raise RuntimeError('This compact runtime requires Windows CPython3.14x64')
    verify_environment()
    for name in ('postgres.exe','initdb.exe','pg_ctl.exe'):
        if not (ROOT/'artifacts/service-postgres-v1/pgsql/bin'/name).is_file():raise RuntimeError('Missing database executable: '+name)
    for name in ('fastapi','uvicorn','psycopg','ezdxf','cv2','PIL','pypdf','pypdfium2','fitz','rapidocr','onnxruntime'):
        importlib.import_module(name)
    from .ocr_engine import _preflight
    _preflight((ROOT/'artifacts/models/ppocr-v5').resolve())
