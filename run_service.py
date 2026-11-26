"""Run the qualified compact Web V32 service without external site-packages."""
from pathlib import Path
import os
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    if (os.name != 'nt' or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 14) or sys.maxsize <= 2**32):
        raise RuntimeError('Windows x64 CPython 3.14 is required. See QUICKSTART.md.')
    if not (root / 'runtime/algorithm-deps/numpy').is_dir():
        raise RuntimeError('Runtime is missing. Run py -3.14 setup_repository.py --runtime-archive <archive.zip> first.')
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    process = subprocess.Popen([sys.executable, '-I', '-S',
                               str(root / 'scripts/start_clean_demo_v1.py'), *sys.argv[1:]], cwd=root, env=env)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Windows console delivers Ctrl+C to the launcher too; let it stop its DB safely.
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                continue


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
