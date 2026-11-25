"""Verify the previously published smoke result survives Compose recreation."""
import hashlib
import json
from pathlib import Path
import urllib.request

record = json.loads(Path('/app/artifacts/profile-service-v1/docker-smoke-result.json').read_text())
base = 'http://127.0.0.1:8000'
with urllib.request.urlopen(base + record['status_url'], timeout=30) as response:
    assert json.load(response)['status'] == 'completed'
with urllib.request.urlopen(base + record['download_url'], timeout=30) as response:
    assert hashlib.sha256(response.read()).hexdigest() == record['dxf_sha256']
print('PASS: document and DXF survived recreation of all containers.')
