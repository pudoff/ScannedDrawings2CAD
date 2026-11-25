"""Build a distinct Linux inventory from the reviewed Windows source payload.

Windows DLLs and PostgreSQL are never copied. Only files listed in the compact
index enter /app; their original hashes are checked before platform adaptation.
The generated Linux index/trust anchor is immutable once the image is built.
"""
import gzip
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def main(source, target):
    source, target = Path(source), Path(target)
    index = json.loads((source / 'runtime/compact/index.json').read_text())
    original_index_sha = digest(source / 'runtime/compact/index.json')
    trust_source = ast.parse((source / 'src/cad_recovery/compact_trust_v1.py').read_text())
    expected_index_sha = next(ast.literal_eval(node.value) for node in trust_source.body
                              if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'INDEX_SHA256' for t in node.targets))
    if original_index_sha != expected_index_sha:
        raise ValueError('Source compact index differs from its trust anchor')
    sources = {p.relative_to(source).as_posix().casefold(): p
               for folder in ('src', 'scripts', 'configs', 'runtime/compact', 'artifacts/models')
               for p in (source / folder).rglob('*') if p.is_file()}
    for row in index['files']:
        if row['kind'] == 'environment':
            continue
        path = sources[row['path'].casefold()]
        data = path.read_bytes()
        # Git checkouts may use LF or CRLF. Recover the exact inventoried bytes.
        variants = [data, data.replace(b'\r\n', b'\n'), data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')]
        data = next((v for v in variants if hashlib.sha256(v).hexdigest() == row['sha256']), None)
        if data is None:
            raise ValueError('Source differs from compact index: ' + row['path'])
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    # An openly redistributable font replaces the Windows-only Cambria face.
    font_source = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font_target = target / 'runtime/font/Fonts' / font_source.name
    font_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(font_source, font_target)
    sys.path.insert(0, str(target / 'runtime/algorithm-deps'))
    from fontTools.ttLib import TTFont
    with TTFont(font_target) as face:
        if not {0x410, 0x44f, 0xd8, 0x2300, 0xb1, 0xb0} <= face.getBestCmap().keys():
            raise ValueError('Font lacks required Cyrillic/engineering glyphs')
    font = dict(file=font_source.name, sha256=digest(font_target), glyph_coverage_verified=True)
    export = target / 'src/cad_recovery/native_annotation_export_v1.py'
    data = export.read_bytes()
    old = b"FONT_FILE = 'cambria.ttc'"
    if data.count(old) != 1:
        raise ValueError('Annotation font adapter no longer matches')
    export.write_bytes(data.replace(old, b"FONT_FILE = 'DejaVuSans.ttf'"))

    # Keep frozen bindings as ancestry, but assign Linux/font-specific identities.
    bindings_path = target / 'runtime/compact/bindings.json.gz'
    bindings = json.loads(gzip.decompress(bindings_path.read_bytes()))
    shutil.copyfile(bindings_path, bindings_path.with_name('windows-bindings.json.gz'))
    schemas = {f'processing-profile/{i}.0' for i in range(1, 10)} | {f'web-table-extension/{i}.0' for i in range(1, 8)}
    originals = {}
    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if {'sha256', 'code', 'schema_version'} <= value.keys():
                if value['schema_version'] not in schemas:
                    raise ValueError('Unknown binding schema: ' + value['schema_version'])
                originals[value['sha256']] = value
            for item in value.values():
                collect(item)
    collect(bindings)
    adapted = {}
    def adapt(value):
        if isinstance(value, list):
            return [adapt(v) for v in value]
        if not isinstance(value, dict):
            return value
        is_binding = value.get('schema_version') in schemas and 'sha256' in value
        if is_binding and value['sha256'] in adapted:
            return adapted[value['sha256']]
        result = {k: adapt(v) for k, v in value.items()}
        if 'annotation_font' in result:
            result['annotation_font'] = font
        if isinstance(result.get('code'), dict):
            for name in result['code']:
                path = target / name
                if path.is_file():
                    result['code'][name] = digest(path)
        if is_binding:
            for key in ('legacy_binding_sha256', 'baseline_binding_sha256'):
                original = result.get(key)
                if original in originals:
                    result[key] = adapt(originals[original])['sha256']
            result['source_windows_binding_sha256'] = result.pop('sha256')
            result['deployment_platform'] = 'linux-amd64-dejavu-v1'
            result['platform_qualification'] = 'Windows qualification is ancestry, not Linux acceptance'
            result['sha256'] = hashlib.sha256(encoded(result)).hexdigest()
            adapted[value['sha256']] = result
        return result
    bindings_path.write_bytes(gzip.compress(encoded(adapt(bindings)), mtime=0))
    shutil.copytree(source / 'docker', target / 'docker')
    for folder in ('artifacts/service-runtime-v1/deps', 'artifacts/cv-skeleton-runtime-v1/deps'):
        (target / folder).mkdir(parents=True, exist_ok=True)

    # Pin all installed Linux packages; installation RECORD files are inventoried too.
    dependencies = {dist.metadata['Name']: dist.version for dist in
                    importlib.metadata.distributions(path=[str(target / 'runtime/algorithm-deps')])}
    trust = 'src/cad_recovery/compact_trust_v1.py'
    files = []
    for folder in ('src', 'scripts', 'configs', 'runtime', 'artifacts/models', 'docker'):
        for path in sorted((target / folder).rglob('*')):
            if not path.is_file() or path.relative_to(target).as_posix() in (trust, 'runtime/compact/index.json'):
                continue
            name = path.relative_to(target).as_posix()
            files.append(dict(path=name, sha256=digest(path), size_bytes=path.stat().st_size,
                              kind='environment' if name.startswith('runtime/algorithm-deps/') else 'runtime'))
    linux_index = {**index, 'files': files, 'dependencies': dependencies, 'font': font,
                   'producer': 'docker-linux-v1', 'source_windows_index_sha256': original_index_sha,
                   'platform': 'linux/amd64', 'python': sys.version,
                   'policy': {**index['policy'], 'compatibility': 'Linux bindings; Windows results require explicit reprocessing',
                              'font_substitution': 'Cambria -> DejaVu Sans; text metrics may differ'}}
    index_path = target / 'runtime/compact/index.json'
    index_path.write_bytes(encoded(linux_index))
    (target / trust).write_text("INDEX_SHA256 = " + repr(digest(index_path)) + '\n')
    print(f'Prepared Linux inventory: {len(files)} files, {len(dependencies)} packages.')


if __name__ == '__main__':
    main(*sys.argv[1:])
