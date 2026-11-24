"""Explicit, checksum-verified ONNX provisioning for P16.2i.1; no inference."""
from pathlib import Path
import hashlib
import urllib.request

from cad_recovery.dataset_layout_v1 import checked, encoded, pin, read

ASSETS = 'artifacts/ocr-model-comparison-assets-v1'
MODELS = {
    'cyrillic_PP-OCRv5_rec_mobile.onnx': '90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711',
    'ch_PP-OCRv5_det_server.onnx': '0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad',
}
MAX_BYTES = 256 * 1024**2


def fetch(root):
    import yaml
    folder = root/ASSETS
    sources = read(folder/'sources.json')
    for record in sources:
        checked(folder, {key: record[key] for key in ('path', 'sha256')})
    manifest = yaml.safe_load((folder/'rapidocr-default-models-v3.9.2.yaml').read_text('utf-8'))
    models = manifest['onnxruntime']['PP-OCRv5']
    records = []
    for filename, sha in MODELS.items():
        role = 'rec' if '_rec_' in filename else 'det'
        advertised = models[role][filename.removesuffix('.onnx')]
        assert advertised['SHA256'] == sha, 'Upstream weight identity changed'
        url = advertised['model_dir']
        assert url.startswith('https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/')
        destination = folder/filename
        if not destination.exists():
            temporary = destination.with_suffix('.partial')
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open('xb') as stream:
                while block := response.read(1024**2):
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError('Download exceeds explicit ONNX size budget')
                    digest.update(block)
                    stream.write(block)
            if digest.hexdigest() != sha:
                raise ValueError('Download SHA mismatch; partial retained for inspection')
            temporary.rename(destination)
        checked(root, {'path': destination.relative_to(root).as_posix(), 'sha256': sha})
        record = {**pin(root, destination.relative_to(root).as_posix()), 'url': url,
                  'size_bytes': destination.stat().st_size, 'role': role}
        records.append(record)
        print({'filename': filename, 'bytes': record['size_bytes'], 'sha256_verified': True}, flush=True)
    output = folder/'models.json'
    payload = encoded({'schema_version': 'ocr-comparison-model-assets/1.0', 'files': records})
    if output.exists():
        assert output.read_bytes() == payload, 'Existing model manifest differs'
    else:
        output.write_bytes(payload)


if __name__ == '__main__':
    fetch(Path(__file__).resolve().parents[1])
