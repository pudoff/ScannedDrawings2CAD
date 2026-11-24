"""Start Web V23 with its existing private runtimes on the import path.

No package installation, profile changes or machine-wide environment changes.
The environment is inherited by the launcher-owned Web and queue processes.
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT/'src', ROOT/'scripts', ROOT/'artifacts/service-runtime-v1/deps',
         ROOT/'artifacts/cv-skeleton-runtime-v1/deps']


def main():
    if not all(path.is_dir() for path in PATHS):
        raise RuntimeError('Existing service/CV runtime directories are missing; inspect local setup.')
    for path in PATHS:
        sys.path.insert(0, str(path))
    prior = os.environ.get('PYTHONPATH')
    os.environ['PYTHONPATH'] = os.pathsep.join([*(str(p) for p in PATHS), *([prior] if prior else [])])
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'artifacts/.cache'))
    runpy.run_path(str(ROOT/'scripts/run_profiles_v23.py'), run_name='__main__')


if __name__ == '__main__':
    main()
