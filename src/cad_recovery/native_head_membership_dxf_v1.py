"""Store complete source-ink membership evidence in DXF OBJECTS only.

This inventory does not authorize a whole CAD arrowhead, change source ink,
or assign an exclusive semantic role.  The existing drawing is replaced only
after its metadata, entity signatures, layers and units survive a clean audit
and save/reopen check.
"""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

import ezdxf
from ezdxf.lldxf.tagwriter import TagCollector

from .native_subcell_export_v1 import entity_signature


ROOT_NAME = 'CADRECOVERY_HEAD_MEMBERSHIP_V1'
SCHEMA_VERSION = 'native-head-membership-dxf/1.0'
REPORT_VERSION = 'native-head-membership/1.0'
POLICY = {'maximum_records': 64, 'maximum_record_payload_bytes': 32 * 1024 * 1024,
          'maximum_total_payload_bytes': 64 * 1024 * 1024,
          'maximum_chunks': 300_000, 'maximum_chunk_characters': 240,
          'maximum_json_depth': 64, 'maximum_json_nodes': 4_000_000,
          'maximum_existing_extension_objects': 100_000}


def _json_text(value):
    """Reject coercions (tuple, non-string key, NaN) before touching a DXF."""
    nodes = 0

    def check(item, depth):
        nonlocal nodes
        nodes += 1
        if depth > POLICY['maximum_json_depth'] or nodes > POLICY['maximum_json_nodes']:
            raise ValueError('Membership JSON structure budget exceeded')
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise ValueError('Finite, losslessly serializable membership JSON required')

    check(value, 0)
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError('Serializable membership JSON required') from exc


def _report_text(report):
    if type(report) is not dict:
        raise ValueError('Complete native head membership report required')
    text = _json_text(report)
    if (report.get('schema_version') != REPORT_VERSION or report.get('complete') is not True
            or type(report.get('source_context_id')) is not str or not report['source_context_id']
            or report.get('limits') != [] or report.get('source_ink_partition_complete') is not True
            or type(report.get('roles_changed')) is not int or report['roles_changed'] != 0):
        raise ValueError('Complete source-bound native head membership report required')
    for key in ('semantic_exclusive_head_confirmed', 'whole_triangle_ownership_confirmed',
                'complete_cad_head_approved', 'cad_modification', 'cad_modification_performed',
                'physical_geometry_approved', 'head_uncertainty_reduced'):
        if report.get(key) is not False:
            raise ValueError('Source evidence only; whole CAD head semantics must remain withheld')
    if (report.get('possible_membership_is_not_exclusive_ownership') is not True
            or report.get('conditional_core_semantics_withheld') is not True):
        raise ValueError('Possible membership cannot authorize exclusive CAD head ownership')
    return text


def _prepare(reports):
    if type(reports) is not list or len(reports) > POLICY['maximum_records']:
        raise ValueError('Membership record count budget exceeded or report list missing')
    rows, identities, size, chunks = [], set(), 0, 0
    for index, report in enumerate(reports):
        report_text = _report_text(report)
        identity = report['source_context_id']
        if identity in identities:
            raise ValueError('Duplicate membership source context identity')
        identities.add(identity)
        digest = sha256(report_text.encode('ascii')).hexdigest()
        envelope = {'schema_version': SCHEMA_VERSION, 'index': index, 'records_count': len(reports),
                    'source_context_id': identity, 'report_sha256': digest, 'report': report}
        payload = _json_text(envelope)
        length = len(payload)
        count = (length + POLICY['maximum_chunk_characters'] - 1) // POLICY['maximum_chunk_characters']
        size += length
        chunks += count
        if (length > POLICY['maximum_record_payload_bytes']
                or size > POLICY['maximum_total_payload_bytes'] or chunks > POLICY['maximum_chunks']):
            raise ValueError('Membership payload or chunk budget exceeded')
        rows.append({'key': f'CHAIN_{index:06d}', 'chain_id': identity,
                     'sha256': digest, 'payload': payload, 'chunks': count})
    return rows, size, chunks


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key in membership record')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Non-finite JSON in membership record: ' + value)


def _read_doc(doc):
    if ROOT_NAME not in doc.rootdict:
        return []
    root = doc.rootdict.get(ROOT_NAME)
    if not hasattr(root, 'dxftype') or root.dxftype() != 'DICTIONARY':
        raise ValueError('Membership root dictionary required')
    entries = list(root.items())
    if not entries or len(entries) > POLICY['maximum_records']:
        raise ValueError('Membership record count budget exceeded or empty dictionary')
    reports, size, count = [], 0, 0
    for index, (name, record) in enumerate(sorted(entries)):
        if name != f'CHAIN_{index:06d}' or not hasattr(record, 'dxftype') or record.dxftype() != 'XRECORD':
            raise ValueError('Ordered per-chain membership XRECORD required')
        pieces, length = [], 0
        for tag in record.tags:
            if (tag.code != 1 or type(tag.value) is not str or not tag.value
                    or len(tag.value) > POLICY['maximum_chunk_characters'] or not tag.value.isascii()):
                raise ValueError('Membership XRECORD requires compact ASCII group-code 1 chunks')
            length += len(tag.value)
            count += 1
            if (length > POLICY['maximum_record_payload_bytes']
                    or size + length > POLICY['maximum_total_payload_bytes']
                    or count > POLICY['maximum_chunks']):
                raise ValueError('Membership payload or chunk budget exceeded')
            pieces.append(tag.value)
        if not pieces:
            raise ValueError('Empty membership record')
        size += length
        payload = ''.join(pieces)
        try:
            row = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError('Invalid membership record JSON') from exc
        if (type(row) is not dict or set(row) != {'schema_version', 'index', 'records_count',
                                                'source_context_id', 'report_sha256', 'report'}
                or row.get('schema_version') != SCHEMA_VERSION
                or type(row.get('index')) is not int or row['index'] != index
                or type(row.get('records_count')) is not int or row['records_count'] != len(entries)):
            raise ValueError('Membership record envelope differs')
        report_text = _report_text(row['report'])
        if (row['source_context_id'] != row['report']['source_context_id']
                or row['report_sha256'] != sha256(report_text.encode('ascii')).hexdigest()
                or _json_text(row) != payload):
            raise ValueError('Membership record SHA256, binding or canonical payload differs')
        reports.append(row['report'])
    # Recheck duplicate source identities and the aggregate serialization budget.
    _prepare(reports)
    return reports


def read(path):
    """Return original reports after validating their bindings and SHA256."""
    return _read_doc(ezdxf.readfile(path))


def _tag_sha(entity, version):
    collector = TagCollector(dxfversion=version)
    entity.export_dxf(collector)
    # Collector emits primitive DXFTag values, including all XDATA and handles.
    return sha256(_json_text([[tag.code, {'bytes_hex': tag.value.hex()} if isinstance(tag.value, bytes)
                                         else tag.value] for tag in collector.tags]).encode('ascii')).hexdigest()


def _extensions(doc):
    """Bind existing provenance XRECORDs as well as their dictionary handles."""
    pending = [entity.get_extension_dict().dictionary for entity in doc.modelspace()
               if entity.has_extension_dict]
    result = {}
    while pending:
        entity = pending.pop()
        if entity.dxf.handle in result:
            continue
        if len(result) >= POLICY['maximum_existing_extension_objects']:
            raise ValueError('Existing extension dictionary verification budget exceeded')
        result[entity.dxf.handle] = _tag_sha(entity, doc.dxfversion)
        if entity.dxftype() in ('DICTIONARY', 'ACDBDICTIONARYWDFLT'):
            for _, child in entity.items():
                if not hasattr(child, 'dxf'):
                    raise ValueError('Existing extension dictionary contains an unresolved object')
                pending.append(child)
        if entity.has_extension_dict:
            pending.append(entity.get_extension_dict().dictionary)
    return result


def _state(doc):
    entities = {entity.dxf.handle: entity_signature(entity) for entity in doc.modelspace()}
    exact = {entity.dxf.handle: _tag_sha(entity, doc.dxfversion) for entity in doc.modelspace()}
    layers = {entity.dxf.handle: _tag_sha(entity, doc.dxfversion) for entity in doc.layers}
    return {'entities': entities, 'exact_entities': exact, 'layers': layers,
            'extensions': _extensions(doc), 'units': doc.units}


def embed(path, reports):
    """Attach bounded evidence to a fresh DXF without emitting CAD arrowheads."""
    rows, size, chunks = _prepare(reports)
    target = Path(path)
    original_digest = sha256(target.read_bytes()).hexdigest()
    doc = ezdxf.readfile(target)
    if ROOT_NAME in doc.rootdict:
        raise ValueError('Membership root already present; refusing to overwrite evidence')
    before = _state(doc)
    audit = doc.audit()
    if audit.errors or audit.fixes or before != _state(doc):
        raise ValueError('Input DXF requires audit repairs or changes')
    proof = {'schema_version': SCHEMA_VERSION, 'records_count': len(rows), 'payload_bytes': size,
        'chunks_count': chunks, 'root_dictionary': ROOT_NAME if rows else None,
        'record_sha256_by_chain': {row['chain_id']: row['sha256'] for row in rows},
        'all_entity_roundtrip_sha256': before['entities'],
        'all_entity_exact_roundtrip_sha256': before['exact_entities'],
        'existing_extension_roundtrip_sha256': before['extensions'],
        'layers_roundtrip_sha256': before['layers'], 'units': before['units'],
        'audit_errors': 0, 'audit_fixes': 0, 'modelspace_changed': False,
        'layers_changed': False, 'units_changed': False, 'xdata_changed': False,
        'provenance_changed': False, 'source_evidence_only': True,
        'complete_cad_head_exported': False, 'file_saved': False}
    if not rows:
        proof['dxf_sha256'] = original_digest
        return proof
    root = doc.rootdict.add_new_dict(ROOT_NAME, hard_owned=True)
    for row in rows:
        payload = row['payload']
        root.add_xrecord(row['key']).extend((1, payload[i:i + POLICY['maximum_chunk_characters']])
            for i in range(0, len(payload), POLICY['maximum_chunk_characters']))
    descriptor, temporary = tempfile.mkstemp(prefix='.head-membership-', suffix='.dxf', dir=target.parent)
    os.close(descriptor)
    try:
        doc.saveas(temporary)
        reopened = ezdxf.readfile(temporary)
        audit = reopened.audit()
        if audit.errors or audit.fixes:
            raise ValueError('Membership DXF failed roundtrip audit')
        if _state(reopened) != before:
            raise ValueError('Membership metadata changed existing modelspace, layers or units')
        if _read_doc(reopened) != reports:
            raise ValueError('Membership DXF report readback differs')
        if sha256(target.read_bytes()).hexdigest() != original_digest:
            raise ValueError('Target DXF changed during metadata verification')
        proof.update(file_saved=True, dxf_sha256=sha256(Path(temporary).read_bytes()).hexdigest())
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return proof
