"""Public provisioning helpers for isolated, hash-locked Windows environments."""
import email
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import venv

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ALLOWED_HOSTS = {'pypi.org', 'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def request(url, *, head=False):
    if urllib.parse.urlsplit(url).hostname not in ALLOWED_HOSTS or not url.startswith('https://'):
        raise ValueError('Unapproved package host')
    return urllib.request.urlopen(urllib.request.Request(url, method='HEAD' if head else 'GET'), timeout=60)


def pypi_wheel(name, version=None):
    url = 'https://pypi.org/pypi/'+name+('/'+version if version else '')+'/json'
    with request(url) as response:
        data = json.load(response)
    priorities = {tag: i for i, tag in enumerate(sys_tags())}
    candidates = []
    for item in data['urls']:
        if not item['filename'].endswith('.whl') or item['yanked']:
            continue
        compatible = [priorities[tag] for tag in parse_wheel_filename(item['filename'])[3] if tag in priorities]
        if compatible:
            candidates.append((min(compatible), item))
    if not candidates:
        raise RuntimeError('No compatible binary wheel: '+name)
    item = min(candidates, key=lambda pair: pair[0])[1]
    return {'name': canonicalize_name(data['info']['name']), 'version': data['info']['version'], 'filename': item['filename'],
            'url': item['url'], 'sha256': item['digests']['sha256'], 'bytes': item['size'],
            'requires_dist': data['info']['requires_dist'] or [], 'metadata_source': url}


def indexed_wheel(index, name, version):
    import re
    with request(index) as response:
        document = response.read().decode('utf-8')
    links = re.findall(r'href="([^"]+)"', document)
    selected = [url for url in links if urllib.parse.unquote(url).split('/')[-1].split('#')[0] == name+'-'+version+'-cp314-cp314-win_amd64.whl']
    if len(selected) != 1 or '#sha256=' not in selected[0]:
        raise RuntimeError('Missing unambiguous official wheel/SHA: '+name)
    url, sha = selected[0].split('#sha256=')
    with request(url, head=True) as response:
        size = int(response.headers['Content-Length'])
    with request(url+'.metadata') as response:
        raw_metadata = response.read()
    metadata = email.message_from_bytes(raw_metadata)
    if metadata['Version'] != version or canonicalize_name(metadata['Name']) != canonicalize_name(name):
        raise ValueError('Official wheel metadata identity mismatch')
    return {'name': name, 'version': version, 'filename': urllib.parse.unquote(url.rsplit('/', 1)[-1]),
            'url': url, 'sha256': sha, 'bytes': size, 'requires_dist': metadata.get_all('Requires-Dist') or [],
            'metadata_source': index, 'wheel_metadata_sha256': hashlib.sha256(raw_metadata).hexdigest()}


def validate_dependencies(records):
    versions = {canonicalize_name(row['name']): row['version'] for row in records}
    if len(versions) != len(records):
        raise ValueError('Repeated package name')
    for row in records:
        for value in row['requires_dist']:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            name = canonicalize_name(requirement.name)
            if name not in versions or versions[name] not in requirement.specifier:
                raise ValueError('Unsatisfied locked dependency: '+row['name']+' -> '+value)


def fetch_wheels(records, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for row in records:
        if Path(row['filename']).name != row['filename'] or not row['filename'].endswith('.whl'):
            raise ValueError('Unsafe wheel filename')
        path = folder/row['filename']
        if not path.exists():
            print('download', row['filename'], round(row['bytes']/1024**2, 1), 'MiB', flush=True)
            partial = path.with_suffix('.partial')
            size = 0
            announced = 0
            with request(row['url']) as response, partial.open('wb') as stream:
                while chunk := response.read(1024**2):
                    stream.write(chunk)
                    size += len(chunk)
                    if size-announced >= 256*1024**2:
                        announced = size
                        print('received', round(size/1024**2), 'MiB', flush=True)
            if size != row['bytes'] or digest(partial) != row['sha256']:
                raise ValueError('Wheel SHA/length mismatch: '+row['filename'])
            partial.replace(path)
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Changed offline wheel: '+row['filename'])


def install_offline(records, wheelhouse, lock, environment, root):
    validate_dependencies(records)
    expected = ''.join(row['name']+'=='+row['version']+' --hash=sha256:'+row['sha256']+'\n' for row in records)
    if lock.read_text('utf-8') != expected:
        raise ValueError('Environment lock changed')
    for row in records:
        if digest(wheelhouse/row['filename']) != row['sha256']:
            raise ValueError('Offline wheel changed')
    if environment.exists() or not environment.resolve().is_relative_to(root/'artifacts/environments'):
        raise ValueError('Choose a new environment under artifacts/environments')
    environment.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(environment)
    python = str(environment/'Scripts/python.exe')
    env = dict(os.environ, PIP_CACHE_DIR=str(root/'pip/cache'), PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--require-hashes', '--find-links', str(wheelhouse), '-r', str(lock)], cwd=root, env=env, check=True)
    subprocess.run([python, '-m', 'pip', 'check'], cwd=root, env=env, check=True)

