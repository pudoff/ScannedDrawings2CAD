"""Freeze and replay source selection before visual OCR-control review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from cad_recovery.clear_pairs import contained, digest
from cad_recovery.family_catalog_v2 import stable_digest
from cad_recovery.label_release_v1 import checked_pin
from cad_recovery.ocr_control_review_v1 import _base as previous_context
from cad_recovery.ocr_release_v3 import _package
from cad_recovery.ocr_control_selection_v2 import PROTOCOL_V2, build_control_queue_v2

PARENT = {'path': 'artifacts/ocr-baseline-final-v2.json',
          'sha256': '6ee85d8133ce9758b590ad891890430ec6f8232e3b20cf3a9efa611d0b744dc6'}
RELEASE = {'path': 'artifacts/ocr-release-v3/manifest.json',
           'sha256': '40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'}
COALLOCATION = {'path': 'artifacts/coallocation-review-v1/manifest.json',
               'sha256': '0bb0ca00ebd2fa766272179cb8f0bfb2debdb988806c6fb30d61caf46b8523c4'}
CODE = ['scripts/prepare_ocr_control_v2.py', *['src/cad_recovery/'+name+'.py' for name in (
    'ocr_control_selection_v2', 'ocr_control_selection_v1', 'ocr_control_review_v1',
    'coallocation_policy_v1', 'ocr_lineage_v1', 'family_catalog_v2', 'ocr_release_v3')]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def pin(root, path):
    return {'path': path, 'sha256': digest(contained(root, path))}


def inputs(root):
    parent = read(checked_pin(root, PARENT))
    require(parent['release'] == RELEASE, 'Previous baseline release changed')
    values, _, _ = previous_context(root)
    release_dir, _ = _package(root, RELEASE)
    coallocation_dir, _ = _package(root, COALLOCATION)
    values['overlay'] = read(coallocation_dir.parent/'overlay.json')
    values['task_splits'] = read(release_dir.parent/'task-splits.json')
    readings = read(coallocation_dir.parent/'readings.json')
    reviewed = [{'path': row['input_path'], 'sha256': row['input_sha256']}
                for packet in readings for row in packet['review']['sources']+packet['review'].get('historical_comparisons', [])]
    require(len(reviewed) == len({row['path'] for row in reviewed}) == 111, 'Previous visual review scope differs')
    values['prior_reviewed_sources'] = reviewed
    return values


def queue(values):
    return build_control_queue_v2(*(values[key] for key in (
        'old_intake', 'new_intake', 'catalog', 'policy', 'handoff', 'exclusions',
        'overlay', 'task_splits', 'prior_reviewed_sources')), protocol=PROTOCOL_V2)


def prepare(root, output):
    target = contained(root, output)
    require(target.parent == root/'configs/reviews' and not target.exists(), 'Use a new protocol path')
    values = inputs(root)
    dependencies = read(root/'configs/reviews/ocr-control-protocol-v1.json')['dependencies']
    protocol = {'schema_version': 'ocr-control-protocol/2.0', 'task': 'P14.1b.7',
        'created_before_canonical_selection_utc': datetime.now(timezone.utc).isoformat(),
        'parent': PARENT, 'release': RELEASE, 'coallocation': COALLOCATION, 'dependencies': dependencies,
        'code': [pin(root, name) for name in CODE], 'protocol': PROTOCOL_V2,
        'prior_reviewed_sources': values['prior_reviewed_sources'],
        'metadata_explored_before_protocol': True, 'images_inspected_in_this_step_before_selection': False,
        'prior_pair_quality_reviews_exist': True, 'OCR_run': False, 'training_started': False,
        'independent_evaluation': False, 'split_assignment': None}
    target.write_bytes(encoded(protocol))
    return pin(root, output)


def checked_protocol(root, record):
    protocol = read(checked_pin(root, record))
    require(isinstance(protocol, dict) and set(protocol) == {'schema_version', 'task',
        'created_before_canonical_selection_utc', 'parent', 'release', 'coallocation', 'dependencies',
        'code', 'protocol', 'prior_reviewed_sources', 'metadata_explored_before_protocol',
        'images_inspected_in_this_step_before_selection', 'prior_pair_quality_reviews_exist',
        'OCR_run', 'training_started', 'independent_evaluation', 'split_assignment'}, 'Unexpected protocol fields')
    require(protocol['schema_version'] == 'ocr-control-protocol/2.0' and protocol['task'] == 'P14.1b.7'
        and protocol['parent'] == PARENT and protocol['release'] == RELEASE and protocol['coallocation'] == COALLOCATION
        and stable_digest(protocol['protocol']) == stable_digest(PROTOCOL_V2), 'Selection scope changed')
    require(protocol['code'] == [pin(root, name) for name in CODE], 'Selection implementation changed')
    require(all(protocol[key] is False for key in ('images_inspected_in_this_step_before_selection', 'OCR_run',
        'training_started', 'independent_evaluation')) and protocol['split_assignment'] is None
        and protocol['metadata_explored_before_protocol'] is True and protocol['prior_pair_quality_reviews_exist'] is True,
        'Invalid selection claims')
    require(protocol['dependencies'] == read(root/'configs/reviews/ocr-control-protocol-v1.json')['dependencies'], 'Previous input pins differ')
    for record in protocol['dependencies'].values():
        checked_pin(root, record)
    values = inputs(root)
    require(protocol['prior_reviewed_sources'] == values['prior_reviewed_sources'], 'Previous review membership changed')
    require(datetime.fromisoformat(protocol['created_before_canonical_selection_utc']).tzinfo is not None, 'Missing protocol timezone')
    return protocol, values


def select(root, protocol_pin, output):
    target = contained(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Use a new selection path')
    protocol, values = checked_protocol(root, protocol_pin)
    result = {'schema_version': 'ocr-control-selection/2.0', 'protocol': protocol_pin,
        'selected_at_utc': datetime.now(timezone.utc).isoformat(), 'queue': queue(values),
        'images_inspected_in_this_step_before_selection': False, 'OCR_run': False, 'training_started': False,
        'independent_evaluation': False, 'split_assignment': None}
    require(datetime.fromisoformat(result['selected_at_utc']) >= datetime.fromisoformat(protocol['created_before_canonical_selection_utc']), 'Selection predates protocol')
    target.write_bytes(encoded(result))
    return pin(root, output)


def verify(root, record):
    selection = read(checked_pin(root, record))
    require(isinstance(selection, dict) and set(selection) == {'schema_version', 'protocol', 'selected_at_utc',
        'queue', 'images_inspected_in_this_step_before_selection', 'OCR_run', 'training_started',
        'independent_evaluation', 'split_assignment'}, 'Unexpected selection fields')
    protocol, values = checked_protocol(root, selection['protocol'])
    require(selection['schema_version'] == 'ocr-control-selection/2.0' and
        stable_digest(selection['queue']) == stable_digest(queue(values)), 'Saved selection differs from recipe')
    require(all(selection[key] is False for key in ('images_inspected_in_this_step_before_selection', 'OCR_run',
        'training_started', 'independent_evaluation')) and selection['split_assignment'] is None, 'Invalid selection result claims')
    selected = datetime.fromisoformat(selection['selected_at_utc'])
    require(selected.tzinfo is not None and selected >= datetime.fromisoformat(protocol['created_before_canonical_selection_utc']), 'Selection time differs')
    return selection['queue']['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', default='configs/reviews/ocr-control-protocol-v2.json')
    p = sub.add_parser('select'); p.add_argument('--protocol', default='configs/reviews/ocr-control-protocol-v2.json')
    p.add_argument('--protocol-sha256', required=True); p.add_argument('--output', default='artifacts/ocr-control-selection-v2.json')
    p = sub.add_parser('verify'); p.add_argument('--selection', default='artifacts/ocr-control-selection-v2.json')
    p.add_argument('--selection-sha256', required=True)
    args = parser.parse_args(); root = args.workspace.resolve()
    if args.command == 'prepare':
        result = prepare(root, args.output)
    elif args.command == 'select':
        result = select(root, {'path': args.protocol, 'sha256': args.protocol_sha256}, args.output)
    else:
        result = verify(root, {'path': args.selection, 'sha256': args.selection_sha256})
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
