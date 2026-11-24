"""Explicit public assets only; strict lengths/hashes, no install or customer input."""
import hashlib
from pathlib import Path
import tarfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .dataset_layout_v1 import checked, inside, read
from .family_catalog_v2 import canonical_sha, require

HOSTS = {'github.com', 'codeload.github.com', 'paddle-model-ecology.bj.bcebos.com'}
MAX_BYTES = 256*1024*1024


def fetch_one(root, asset, *, verify_only=False, opener=urlopen):
    canonical_sha(asset['sha256'])
    url = urlparse(asset['url'])
    require(url.scheme == 'https' and url.hostname in HOSTS and not url.username and not url.password,
            'Unsupported public asset URL')
    require(type(asset['bytes']) is int and 0 < asset['bytes'] <= MAX_BYTES, 'Invalid public asset length')
    target = inside(root, asset['path'])
    if target.exists() or verify_only:
        checked(root, {'path': asset['path'], 'sha256': asset['sha256']})
        require(target.stat().st_size == asset['bytes'], 'Asset size mismatch')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name+'.part')
    require(not part.exists(), 'Previous partial download retained; inspect it before retrying')
    sha, count = hashlib.sha256(), 0
    with opener(Request(asset['url'], headers={'User-Agent': 'cad-recovery-public-assets/1.0'}), timeout=60) as response, part.open('xb') as stream:
        declared = response.headers.get('Content-Length')
        if declared is not None: require(int(declared) == asset['bytes'], 'Unexpected HTTP Content-Length')
        while block := response.read(1024*1024):
            count += len(block)
            require(count <= asset['bytes'], 'Download exceeds pinned length')
            sha.update(block); stream.write(block)
    require(count == asset['bytes'], 'Incomplete download; partial retained')
    require(sha.hexdigest() == asset['sha256'], 'Download SHA-256 mismatch; partial retained')
    # Publish without replacing even a concurrently created destination.
    with target.open('xb') as output, part.open('rb') as source:
        while block := source.read(1024*1024): output.write(block)
    # Keep the verified .part as download evidence; it is never used for inference.


def extract_one(root, archive_name, output):
    target = inside(root, output)
    require(not target.exists(), 'Extraction destination exists; never overwritten')
    with tarfile.open(inside(root, archive_name), 'r:gz') as archive:
        names = set()
        for member in archive.getmembers():
            inside(target, member.name)
            require(member.isfile() or member.isdir(), 'Linked or special archive member')
            name = member.name.casefold().rstrip('/')
            require(name not in names, 'Repeated archive member')
            names.add(name)
        target.mkdir(parents=True)
        archive.extractall(target, filter='data')


def fetch_assets(workspace, manifest='configs/ocr-training-assets-v1.json', *, verify_only=False, extract=False):
    root = Path(workspace).resolve(); config = read(inside(root, manifest))
    require(config['schema_version'] == 'ocr-training-public-assets/1.0', 'Unknown asset manifest')
    names = [a['path'].casefold() for a in config['assets']]
    require(names and len(names) == len(set(names)), 'Missing or repeated assets')
    require(not (verify_only and extract), 'Choose verification or extraction')
    for asset in config['assets']:
        fetch_one(root, asset, verify_only=verify_only)
        if extract and asset.get('extract_to') is not None:
            extract_one(root, asset['path'], asset['extract_to'])
    return {'assets_verified': len(names), 'verify_only': verify_only, 'extracted': extract,
            'packages_installed': False, 'customer_inputs_read': False}
