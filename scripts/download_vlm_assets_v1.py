"""Download one fixed public Qwen revision, verify every file, no model calls."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/vlm-assets-v1'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
WEIGHTS = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'


def main():
    api = json.loads((FOLDER / 'model-api.json').read_text('utf-8'))
    assert api['sha'] == REVISION and api['cardData']['license'] == 'apache-2.0'
    files = [r for r in api['siblings'] if r['rfilename'] != '.gitattributes']
    records = []
    for row in files:
        name = row['rfilename']; path = FOLDER / name
        assert '/' not in name and '\\' not in name
        url = 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/' + REVISION + '/' + name
        if not path.exists():
            temporary = path.with_name(path.name + '.part'); started = time.monotonic(); last = started; total = 0
            with urllib.request.urlopen(url, timeout=90) as source, temporary.open('wb') as target:
                while data := source.read(8 * 1024**2):
                    target.write(data); total += len(data)
                    if time.monotonic() - last > 15:
                        print(name, round(total / 1024**2), 'MiB downloaded', flush=True); last = time.monotonic()
            temporary.replace(path)
        with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if name == 'model.safetensors': assert sha == WEIGHTS and sha == row['lfs']['sha256']
        records.append({'path': name, 'sha256': sha, 'url': url, 'size_bytes': path.stat().st_size})
        print('verified', name, path.stat().st_size, sha, flush=True)
    license_url = 'https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/LICENSE'
    path = FOLDER / 'LICENSE'
    if not path.exists():
        with urllib.request.urlopen(license_url, timeout=40) as source: path.write_bytes(source.read())
    records.append({'path': 'LICENSE', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'url': license_url})
    (FOLDER / 'downloads.json').write_text(json.dumps({'model_id': 'Qwen/Qwen3-VL-2B-Instruct',
        'revision': REVISION, 'files': records}, sort_keys=True, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
