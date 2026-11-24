"""Current control coverage, literal placeholders, physical lines and fields."""
from collections import Counter, defaultdict

from .family_catalog_v2 import require, stable_digest
from .ocr_baseline_metrics_v1 import aggregate_scores_v1, normalize_text_v1
from .ocr_baseline_metrics_v3 import score_control_v3, SPLITS


def build_metadata_v4(labels, registry, dispositions, rows):
    slots = labels['new_field_ledger']
    accepted = {r['id']: r for slot in slots for r in slot['lines']}
    require(len({s['id'] for s in slots}) == len(slots) and len(accepted) == sum(len(s['lines']) for s in slots), 'Duplicate field or line')
    require({r['id'] for r in rows} == set(accepted) and len(rows) == len(accepted), 'Control line inventory differs')
    assignments = {r['id']: r for r in registry['assignments']}
    require(len(assignments) == len(registry['assignments'])
            and {r['id'] for r in registry['assignments'] if r['split'] in SPLITS} == set(accepted), 'Registry control inventory differs')
    sources = {r['input']: r for r in dispositions}
    require(len(sources) == len(dispositions), 'Repeated disposition source')
    slot_for_line = {r['id']: slot for slot in slots for r in slot['lines']}
    strata = []
    for row in rows:
        slot, line = slot_for_line[row['id']], accepted[row['id']]
        source = sources[slot['source_path']]
        require(stable_digest({k: v for k, v in row.items() if k != 'image_path'}) == stable_digest(assignments[row['id']])
                and row['text'] == line['text'] and row['image_sha256'] == line['image']['sha256']
                and row['source_sha256'] == slot['source_sha256'] == source['input_sha256']
                and row['source_path'] == slot['source_path'] and row['split'] == slot['split'] == source['declared_split']
                and row['group_id'] == source['group_id'] and row['split'] in SPLITS
                and row['field'] == slot['field'] and row['slot_id'] == slot['id']
                and row['line_index'] == line['line_index']
                and row['name_is_identifier_placeholder'] == slot.get('name_is_identifier_placeholder', False), 'Control reference or routing changed')
        placeholder = row['name_is_identifier_placeholder']
        strata.append({'id': row['id'], 'source_id': row['source_id'], 'source_path': row['source_path'],
                       'split': row['split'], 'role': row['field'], 'slot_id': slot['id'], 'line_index': row['line_index'],
                       'slot_line_count': len(slot['lines']), 'coallocation_group_id': row['group_id'],
                       'image_sha256': row['image_sha256'], 'name_is_identifier_placeholder': placeholder,
                       'text_kind': 'drawing_id' if row['field'] == 'drawing_id' else 'name_identifier_placeholder' if placeholder else 'name_literal'})
    coverage, by_source = {}, {}
    selected = {s['source_path'] for s in slots}
    for path in sorted(selected):
        fields = [s for s in slots if s['source_path'] == path]
        require(len(fields) == 2 and {s['field'] for s in fields} == {'drawing_id', 'part_name'}, 'Incomplete two-field source ledger')
        for field in fields:
            require(field['decision'] in ('accepted_for_ocr_transcription', 'unresolved')
                    and field['split'] == sources[path]['declared_split'], 'Changed field decision or split')
            if field['decision'] == 'unresolved':
                require(field['text'] is None and field['lines'] == [], 'Uncertain field has an OCR target')
            else:
                require(field['lines'] and field['text'] == ' '.join(r['text'] for r in field['lines']), 'Field differs from line texts')
        by_source[path] = {'split': sources[path]['declared_split'], 'field_slots': 2,
                           'decisions': dict(Counter(f['decision'] for f in fields)),
                           'accepted_lines': sum(len(f['lines']) for f in fields)}
    for split in SPLITS:
        fields = [s for s in slots if s['split'] == split]
        part = [r for r in strata if r['split'] == split]
        all_sources = [r for r in dispositions if r['declared_split'] == split]
        coverage[split] = {'available_source_count': len(all_sources), 'selected_source_count': sum(r['split'] == split for r in by_source.values()),
                           'sources_with_accepted_lines': len({r['source_path'] for r in part}),
                           'pending_source_count': sum(r['input'] not in selected for r in all_sources),
                           'field_slots': len(fields), 'decisions': dict(Counter(s['decision'] for s in fields)),
                           'accepted_line_count': len(part), 'unique_image_count': len({r['image_sha256'] for r in part}),
                           'accepted_group_count': len({r['coallocation_group_id'] for r in part}),
                           'placeholder_name_slots': sum(s.get('name_is_identifier_placeholder', False) for s in fields),
                           'multiline_field_count': sum(len(s['lines']) > 1 for s in fields)}
    owners, images = {}, defaultdict(list)
    for row in strata:
        images[row['image_sha256']].append(row['id'])
        for kind, value in (('image', row['image_sha256']), ('source', row['source_id']), ('group', row['coallocation_group_id'])):
            require((kind, value) not in owners or owners[kind, value] == row['split'], 'Cross-split control leakage')
            owners[kind, value] = row['split']
    return {'schema_version': 'ocr-control-metadata/4.0', 'strata': strata,
            'coverage_by_split': coverage, 'coverage_by_source': by_source,
            'field_slot_count': len(slots), 'decision_counts': dict(Counter(s['decision'] for s in slots)),
            'field_ledger': slots, 'unique_image_representative_ids': sorted(min(ids) for ids in images.values()),
            'duplicates': [{'image_sha256': sha, 'ids': sorted(ids)} for sha, ids in sorted(images.items()) if len(ids) > 1],
            'primary_policy': 'all physical accepted lines, including duplicate crops; unresolved fields have no targets',
            'unique_image_policy': 'secondary diagnostic: smallest frozen ID per image hash',
            'independent_evaluation': False}


def score_control_v4(predictions, metadata):
    scores = score_control_v3(predictions, metadata)
    scores['schema_version'] = 'ocr-control-evaluation-metrics/4.0'
    strata = {r['id']: r for r in metadata['strata']}
    for row in predictions:
        m = strata[row['id']]
        require(row['image_sha256'] == m['image_sha256'], 'Prediction image differs from frozen metadata')
    def score(rows):
        return aggregate_scores_v1([{k: r[k] for k in ('id', 'source_id', 'reference', 'prediction', 'status')} for r in rows])
    for split in SPLITS:
        rows = [r for r in predictions if r['split'] == split]
        part = scores['by_split'][split]
        part['by_text_kind'] = {kind: score([r for r in rows if strata[r['id']]['text_kind'] == kind])
                                for kind in sorted({strata[r['id']]['text_kind'] for r in rows})}
        part['by_line_context'] = {kind: score([r for r in rows if (strata[r['id']]['slot_line_count'] > 1) == multi])
                                   for kind, multi in [('single_line_field', False), ('multiline_field', True)]
                                   if any((strata[r['id']]['slot_line_count'] > 1) == multi for r in rows)}
        fields = []
        for slot in metadata['field_ledger']:
            if slot['split'] != split or slot['decision'] != 'accepted_for_ocr_transcription':
                continue
            lines = sorted([r for r in rows if strata[r['id']]['slot_id'] == slot['id']], key=lambda r: strata[r['id']]['line_index'])
            require([r['id'] for r in lines] == [r['id'] for r in slot['lines']], 'Incomplete physical lines for field')
            fields.append({'id': slot['id'], 'role': slot['field'], 'source_path': slot['source_path'],
                           'line_ids': [r['id'] for r in lines], 'line_statuses': [r['status'] for r in lines],
                           'raw_exact': all(r['status'] == 'ok' and r['reference'] == r['prediction'] for r in lines),
                           'normalized_exact': all(r['status'] == 'ok' and normalize_text_v1(r['reference']) == normalize_text_v1(r['prediction']) for r in lines)})
        part['full_field_diagnostic'] = {'accepted_field_count': len(fields),
                                        'raw_exact_fields': sum(r['raw_exact'] for r in fields),
                                        'normalized_exact_fields': sum(r['normalized_exact'] for r in fields), 'fields': fields,
                                        'policy': 'field exact only when every physical line is exact; errors/empties fail; unresolved slots excluded and reported as coverage'}
    return scores
