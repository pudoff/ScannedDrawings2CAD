"""Install the pinned Windows runtime using only the Python standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit
import venv
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Invalid runtime path: ' + str(name))
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Runtime path escapes repository: ' + name)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('Linked runtime path: ' + name)
    return path


def matches(path, row):
    return path.is_file() and path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256']


def payload(root):
    distribution = json.loads((root / 'runtime-distribution.json').read_text(encoding='utf-8'))
    index_path = root / 'runtime/compact/index.json'
    if sha(index_path) != distribution['index_sha256']:
        raise ValueError('Runtime index differs from this distribution')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    rows = [r for r in index['files'] if r['kind'] == 'environment' or r['path'].startswith('artifacts/models/')]
    return distribution, index, rows


def install_archive(root, archive, spec, rows):
    if not matches(archive, spec):
        raise ValueError('Runtime archive SHA256/size differs: ' + str(archive))
    expected = {r['path']: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate runtime path')
    # Validate and stage the complete archive before replacing any installed file.
    with tempfile.TemporaryDirectory(prefix='.setup-', dir=root) as folder:
        stage = Path(folder)
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            names = [e.filename for e in entries]
            if len(set(n.casefold() for n in names)) != len(names) or set(names) != set(expected):
                raise ValueError('Archive file list differs from the pinned payload')
            for entry in entries:
                row = expected[entry.filename]
                if entry.file_size != row['size_bytes'] or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError('Invalid archive entry: ' + entry.filename)
                destination = inside(stage, entry.filename)
                inside(root, entry.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if not matches(destination, row):
                    raise ValueError('Extracted file differs: ' + entry.filename)
        for name in expected:
            destination = inside(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, destination)


def require_python():
    if (os.name != 'nt' or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 14) or sys.maxsize <= 2**32):
        raise RuntimeError('Use Windows x64 CPython 3.14: py -3.14 setup_repository.py ...')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--runtime-archive', type=Path)
    source.add_argument('--runtime-url', help='HTTPS URL of the matching release asset')
    args = parser.parse_args()
    require_python()
    distribution, index, rows = payload(ROOT)
    font = index.get('font')
    if font:
        path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / font['file']
        if not path.is_file() or sha(path) != font['sha256']:
            raise RuntimeError('Required Windows CAD font differs: ' + str(path)
                               + '. See QUICKSTART.md; no font is downloaded or replaced.')
    missing = [r for r in rows if not matches(inside(ROOT, r['path']), r)]
    if missing:
        spec = distribution['archive']
        archive = args.runtime_archive or ROOT / 'release-assets' / spec['filename']
        if args.runtime_url:
            if urlsplit(args.runtime_url).scheme != 'https':
                raise ValueError('Runtime URL must use HTTPS')
            with tempfile.TemporaryDirectory(prefix='.download-', dir=ROOT) as directory:
                downloaded = Path(directory) / spec['filename']
                print('Downloading the pinned runtime archive...', flush=True)
                with urllib.request.urlopen(args.runtime_url, timeout=120) as response, downloaded.open('xb') as output:
                    remaining = spec['size_bytes']
                    while chunk := response.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError('Runtime download exceeds the declared size')
                        output.write(chunk)
                install_archive(ROOT, downloaded, spec, rows)
        elif archive.is_file():
            install_archive(ROOT, archive.resolve(), spec, rows)
        else:
            raise RuntimeError('Runtime payload is absent after git clone. Obtain '
                               + spec['filename'] + ' from the project release or maintainer, then run:\n'
                               + 'py -3.14 setup_repository.py --runtime-archive C:\\Downloads\\' + spec['filename'])
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.exists():
        print('Creating .venv with CPython 3.14...', flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    subprocess.run([str(python), '-I', '-c',
                    'import sys; assert sys.version_info[:2] == (3,14) and sys.maxsize > 2**32'], check=True)
    print('Checking code, libraries, OCR models and PostgreSQL...', flush=True)
    subprocess.run([str(python), '-B', str(ROOT / 'run_service.py'), '--check'], cwd=ROOT, check=True)
    print('READY. Start: .\\.venv\\Scripts\\python.exe run_service.py --open', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print('SETUP FAILED: ' + str(error), file=sys.stderr)
        sys.exit(1)
