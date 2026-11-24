"""Select metadata before pixels; freeze the real-chain diagnostic data boundary."""
import json
from pathlib import Path

from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.hard_raster_runs_v1 import screen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/real-chain-audit-v1'
PREVIOUS = {'path': 'artifacts/head-junction-release-v1.json',
            'sha256': '0a804a4d704e8e236bf778b68cd41961522b2fafbe28bb0c034c3899f0d744e8'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def select(root):
    """No image decoder, model or target CAD file is called by selection."""
    previous = read(checked(root, PREVIOUS))
    def protected(name):
        return {'path': name, 'sha256': (previous['prior_protected_pins'] | previous['files'])[name]}
    initial = protected('artifacts/head-junction-profile-v1/plan.json')
    plan = read(checked(root, initial))
    old = [j for j in plan['jobs'] if j['case']['scope'] == 'partial_real_reviewed_ink']
    selection_seed = pin(root, 'configs/reviews/pixel-role-expansion-selection-v1.json')
    selection = read(checked(root, selection_seed))
    values = [read(checked(root, p)) for p in selection['inputs']]
    groups = read(checked(root, selection['groups']))['groups']
    members = {p: g for g in groups for p in g['members']}
    entries = {e['input']: e for e in values[1]['entries']}
    old_groups = {j['case']['group_id'] for j in old}
    old_paths = {j['case']['source']['path'][len('dataset/'):] for j in old}
    fresh = [{'source_path': s['input'], 'source_sha256': s['input_sha256']}
             for s in values[4]['selected_sources']]
    pools = {'train': {}, 'validation': {}}
    exclusions = {}
    for source in sorted(values[0], key=lambda s: (s['input_sha256'], s['input'])):
        split = source['declared_split']; group = members[source['input']]
        reason = None
        if split not in pools or source['source_kind'] != 'png': reason = 'split_or_format'
        elif source['input'] in old_paths or group['id'] in old_groups: reason = 'used_source_or_group'
        elif group['cross_split'] or group['declared_splits'] != [split]: reason = 'cross_split_group'
        elif entries[source['input']]['visual_status'] not in ('verified', 'inherited_verified'): reason = 'unverified_visual_pair'
        else:
            boundary = screen([{'id': 'candidate', 'declared_split': split, 'input': source['input'],
                                'source_sha256': source['input_sha256']}], groups, values[2:4], fresh)[0]
            if boundary['blocked']: reason = 'closed_test_or_OCR_holdout'
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else: pools[split].setdefault(group['id'], source)
    selected = [{'id': 'real-chain-new-validation-' + str(i), 'group_id': group,
                 'source_record': source, 'independent_evaluation': False}
                for i, (group, source) in enumerate(list(pools['validation'].items())[:2], 1)]
    return {'task': 'P10.2c.3b.2e', 'previous': PREVIOUS, 'previous_plan': initial, 'selection_seed': selection_seed,
            'metadata': selection['inputs'] + [selection['groups']], 'old_jobs': old,
            'available_new_groups': {k: len(v) for k, v in pools.items()}, 'exclusions': exclusions,
            'new_sources': selected, 'selection': 'first two new validation groups by minimum source SHA/path; no readability replacement',
            'old_cases_are_development': True, 'family_independence_confirmed': False,
            'physical_geometry_approved': False, 'closed_test_images_opened': False,
            'model_or_threshold_changes': False, 'training': False,
            'label_scope': 'assistant-reviewed local heads, shafts, extensions and feet; partial pixel roles; no physical targets',
            'protocol': {'max_head_tip_error_native_px': 8., 'max_head_direction_error_deg': 20.,
                         'head_matching': 'one-to-one maximum cardinality, then minimum geometric error; explicit search boxes',
                         'chain_matching': 'both target heads, shaft and reviewed extension/foot routes; unmatched proposals not called false outside reviewed scope',
                         'role_evaluation': 'partial labels only; unknown GT excluded, unknown prediction is missed coverage',
                         'new_profile_admission': False, 'full_role_macro_F1_min': .90,
                         'next_experiment_chosen_from': 'train-origin development losses only'}}


def prepare():
    assert not BASE.exists(), 'Choose a new version; previous evidence is immutable'
    result = select(ROOT)
    result['generator'] = pin(ROOT, 'scripts/prepare_real_chain_audit_v1.py')
    write(BASE / 'selection.json', result)
    # Only now may display assets be opened. The selector itself used metadata only.
    from PIL import Image, ImageDraw
    items = [(j['case']['id'], checked(ROOT, j['views'][0]['gray'])) for j in result['old_jobs']]
    for s in result['new_sources']:
        asset = next(p for p in s['source_record']['prepared_assets'] if p['path'].endswith('/overview.png'))
        items.append((s['id'], checked(ROOT, asset)))
    sheet = Image.new('RGB', (1200, 300 * ((len(items) + 3) // 4)), 'white'); draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(items):
        image = Image.open(path).convert('RGB'); image.thumbnail((294, 272))
        x, y = (i % 4) * 300, (i // 4) * 300
        sheet.paste(image, (x, y + 22)); draw.text((x + 2, y + 2), label, fill='black')
    sheet.save(BASE / 'source-only-contact-sheet.png')
    print(json.dumps({'selection': pin(ROOT, 'artifacts/real-chain-audit-v1/selection.json'),
                      'groups': result['available_new_groups'], 'sources': [s['source_record']['input'] for s in result['new_sources']]}, ensure_ascii=False))


if __name__ == '__main__':
    prepare()
