"""Inspect referenced native resources without opening their declared paths.

Filename matches are discovery candidates only. Embedded OLE bytes and magic
sequences do not establish a renderable presentation, placement or a restored
reference. This module never extracts OLE objects or approves training labels.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

MAX_REGISTERED_FILES = 100_000
MAX_OLE_PAYLOAD_BYTES = 16 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inventory(records):
    _require(isinstance(records, list) and len(records) <= MAX_REGISTERED_FILES,
             'Expected a bounded registered file inventory')
    result, seen = [], set()
    for record in records:
        _require(isinstance(record, dict), 'Invalid registered file')
        path, sha, size = (record.get(key) for key in ('path', 'sha256', 'size_bytes'))
        _require(isinstance(path, str) and bool(path) and '\\' not in path
                 and ':' not in path and '\x00' not in path
                 and not PurePosixPath(path).is_absolute()
                 and all(part not in ('', '.', '..') for part in path.split('/')),
                 'Registered file path must be a canonical relative path')
        _require(path.casefold() not in seen, 'Repeated registered file path')
        seen.add(path.casefold())
        _require(isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha) is not None
                 and type(size) is int and size >= 0, 'Invalid registered file fingerprint')
        result.append({'path': path, 'sha256': sha, 'size_bytes': size})
    return sorted(result, key=lambda row: (row['path'].casefold(), row['path']))


def _dimensions(value):
    try:
        values = list(value)
    except TypeError as error:
        raise ValueError('Expected two image dimensions') from error
    _require(len(values) == 2 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
             'Expected two finite positive image dimensions')
    return [float(v) for v in values]


def _name_candidates(filename, inventory):
    # PureWindowsPath only splits strings; it never resolves, stats or opens a path.
    declared = PureWindowsPath(filename)
    basename = declared.name
    if not basename:
        return basename, []
    parts = [part.casefold() for part in declared.parts]
    result = []
    for record in inventory:
        local_parts = [part.casefold() for part in PurePosixPath(record['path']).parts]
        if local_parts[-1] != basename.casefold():
            continue
        suffix = 0
        for a, b in zip(reversed(parts), reversed(local_parts)):
            if a != b:
                break
            suffix += 1
        result.append({**record, 'matched_suffix_components': suffix, 'identity_verified': False})
    return basename, result


def inspect_native_resource(doc, observation, registered_files):
    """Return unapproved facts for one observed native IMAGE or OLE2FRAME.

The caller owns visibility/instance evidence and verifies the original file's
SHA. A missing INSERT path is allowed for a previously captured pre-copy skip;
this function locates only the native handle and never fabricates an instance.
An inconsistent identity or binary length is refused rather than silently fixed.
"""
    _require(isinstance(observation, dict), 'Expected a native resource observation')
    kind = observation.get('entity_type')
    handle, owner = observation.get('native_handle'), observation.get('native_owner_handle')
    _require(kind in ('IMAGE', 'OLE2FRAME') and isinstance(handle, str) and bool(handle)
             and isinstance(owner, str) and bool(owner), 'Missing native resource identity')
    entity = doc.entitydb.get(handle)
    _require(entity is not None and entity.is_alive and entity.dxftype() == kind
             and entity.dxf.get('handle') == handle and entity.dxf.get('owner') == owner,
             'Resource identity differs from the native drawing')
    inventory = _inventory(registered_files)
    facts = {'resource_kind': 'image' if kind == 'IMAGE' else 'ole', 'entity_type': kind,
             'native_handle': handle, 'native_owner_handle': owner,
             'restored': False, 'resource_rendering_verified': False, 'source_modified': False}
    if kind == 'IMAGE':
        definition = entity.image_def
        _require(definition is not None and definition.is_alive, 'Native IMAGE definition is absent')
        def_handle = entity.dxf.get('image_def_handle')
        filename = definition.dxf.get('filename', '')
        _require(definition.dxftype() == 'IMAGEDEF' and definition.dxf.get('handle') == def_handle
                 and observation.get('image_def_handle') == def_handle,
                 'IMAGE definition identity differs from the observation')
        _require(isinstance(filename, str) and '\x00' not in filename
                 and 'declared_filename' in observation and observation['declared_filename'] == filename,
                 'IMAGE filename differs from the observation')
        dimensions = _dimensions([entity.dxf.image_size.x, entity.dxf.image_size.y])
        _require(_dimensions(observation.get('image_size_px')) == dimensions,
                 'IMAGE dimensions differ from the observation')
        if 'show_image' in observation:
            _require(type(observation['show_image']) is bool
                     and observation['show_image'] == bool(int(entity.dxf.flags) & 1),
                     'IMAGE display flag differs from the observation')
        basename, matches = _name_candidates(filename, inventory)
        facts.update(image_def_handle=def_handle, declared_filename=filename,
                     declared_basename=basename, image_size_px=dimensions,
                     image_def_size_px=_dimensions([definition.dxf.image_size.x, definition.dxf.image_size.y]),
                     registered_name_candidates=matches,
                     reason=('empty_image_reference' if not filename.strip() else
                             'registered_name_candidates_unverified' if matches else
                             'missing_registered_original'))
        return facts

    tags = entity.acdb_ole2frame or []
    chunks = [tag.value for tag in tags if tag.code == 310]
    _require(all(isinstance(chunk, bytes) for chunk in chunks), 'OLE data contains nonbinary chunks')
    length = sum(len(chunk) for chunk in chunks)
    _require(length <= MAX_OLE_PAYLOAD_BYTES, 'OLE payload exceeds the inspection byte limit')
    declared = [tag.value for tag in tags if tag.code == 90]
    _require(len(declared) <= 1 and (not declared or type(declared[0]) is int and declared[0] >= 0),
             'Invalid OLE declared byte length')
    declared_size = declared[0] if declared else None
    _require(declared_size is None or declared_size == length, 'OLE declared length differs from native bytes')
    payload = entity.binary_data()
    _require(payload == b''.join(chunks), 'OLE binary extraction differs from native tags')
    offsets = {}
    for name, magic in (('compound_file', bytes.fromhex('d0cf11e0a1b11ae1')),
                        ('png', bytes.fromhex('89504e470d0a1a0a'))):
        offset = payload.find(magic)
        offsets[name] = offset if offset >= 0 else None
    facts.update(reason='embedded_ole_rendering_unverified' if payload else 'empty_ole_payload',
                 payload={'size_bytes': length, 'sha256': hashlib.sha256(payload).hexdigest(),
                          'declared_size_bytes': declared_size,
                          'declared_size_matches': None if declared_size is None else True,
                          'signature_offsets': offsets,
                          'signature_detection': 'first_magic_only_not_format_validation'})
    return facts
