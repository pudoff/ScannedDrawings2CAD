"""Fixed metadata strata and inherited-line regression for OCR development data.

All text scoring delegates to the frozen V1 comparison policy. This module is
pure: callers authenticate labels, image bytes and prediction provenance. It
does not infer semantic families, choose parameters or approve evaluation data.
"""
from copy import deepcopy
import json

from .family_catalog_v2 import canonical_sha, relative_path
from .ocr_baseline_metrics_v1 import aggregate_scores_v1


ROLE_GROUPS_V2 = {
    **{role: 'title' for role in ('title_designation', 'title_name', 'title_part_name',
                                'related_designation', 'title_related_designation')},
    **{role: 'view_label' for role in ('section_label', 'detail_label', 'detail_view_label_and_printed_scale')},
    **{role: 'roughness' for role in ('roughness_text', 'roughness_notation')},
    **{role: 'quantity' for role in ('hole_count_text', 'hole_quantity', 'chamfer_quantity')},
    **{role: 'table' for role in ('table_column_header', 'table_length_symbol', 'table_length_value',
                                'table_mass_value', 'table_variant_designation')},
    **{role: 'dimension' for role in ('angular_dimension_text', 'angular_nominal', 'chamfer_notation',
        'diameter_dimension_text', 'horizontal_dimension_text', 'linear_dimension_text', 'linear_nominal',
        'lower_deviation_text', 'parenthesized_dimension_text', 'reference_dimension_text',
        'symmetric_tolerance_text', 'thread_notation', 'upper_deviation_text', 'vertical_dimension_text',
        'vertical_reference_dimension', 'vertical_symmetric_deviation_notation')},
}
INHERITED_KINDS_V2 = frozenset({'number', 'reference_number', 'chamfer', 'radius', 'diameter', 'thread', 'unresolved'})
COHORTS_V2 = ('inherited32', 'new58')
METADATA_FIELDS = {'id', 'cohort', 'source_id', 'role', 'role_group'}
PREDICTION_FIELDS = {'id', 'source_id', 'reference', 'prediction', 'status', 'error',
                     'detections', 'order', 'image', 'image_sha256'}
REGRESSION_FIELDS_V2 = ('prediction', 'status', 'error', 'detections', 'order')
SCORE_FIELDS = ('id', 'source_id', 'reference', 'prediction', 'status')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields, message):
    _require(isinstance(value, dict) and fields <= set(value), message)


def _identity(value):
    _require(isinstance(value, str) and bool(value.strip()), 'Expected a nonempty identity')
    return value


def _index(rows, name, *, nonempty=True):
    _require(isinstance(rows, list) and (bool(rows) or not nonempty), 'Expected '+name+' list')
    result = {}
    for row in rows:
        _fields(row, {'id'}, 'Missing '+name+' identity')
        identifier = _identity(row['id'])
        _require(identifier not in result, 'Duplicate '+name+' identity')
        result[identifier] = row
    return result


def _pin(record):
    _require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Expected an exact image/source pin')
    relative_path(record['path']); canonical_sha(record['sha256'])
    return record


def _source_map(labels, cohort):
    inherited = cohort == 'inherited32'
    rows = labels['inherited_source_records' if inherited else 'new_source_records']
    result = _index(rows, cohort+' sources')
    seen_paths = set()
    bindings = {}
    for identifier, row in result.items():
        if inherited:
            _fields(row, {'source'}, 'Missing inherited source pin')
            record = _pin(row['source'])
            _require(record['path'].startswith('dataset/'), 'Inherited source is not dataset-relative')
            path, sha = record['path'].removeprefix('dataset/'), record['sha256']
        else:
            _fields(row, {'input_path', 'input_sha256'}, 'Missing new source binding')
            path, sha = relative_path(row['input_path']), canonical_sha(row['input_sha256'])
        _require(path not in seen_paths, 'Aliased source path')
        seen_paths.add(path)
        bindings[identifier] = {'path': path, 'sha256': sha}
    return bindings


def build_strata_v2(labels_v3, train_records):
    """Build a complete, deterministic partition from authenticated label metadata.

    The cohort names identify this experiment's frozen 32/58 split; exact counts
    are enforced by its wrapper. Small analytical fixtures can exercise the same
    rules. No reference or prediction content is used to choose a role group.
    """
    fields = {'schema_version', 'task', 'inherited_samples', 'new_samples', 'inherited_annotations',
              'inherited_source_records', 'new_source_records', 'unresolved_candidates'}
    _fields(labels_v3, fields, 'Missing V3 label inventories')
    _require(labels_v3['schema_version'] == 'ocr-labels/3.0' and labels_v3['task'] == 'ocr_recognition', 'Expected OCR labels V3')
    inherited = _index(labels_v3['inherited_samples'], 'inherited samples')
    new = _index(labels_v3['new_samples'], 'new samples')
    _require(not set(inherited).intersection(new), 'Inherited/new sample overlap')
    records = _index(train_records, 'train records')
    _require(set(records) == set(inherited)|set(new), 'Train registry does not cover the complete label inventory')
    unresolved = _index(labels_v3['unresolved_candidates'], 'unresolved candidates', nonempty=False)
    _require(not set(unresolved).intersection(records), 'Unresolved candidate has a training record')
    for row in unresolved.values():
        _fields(row, {'text', 'decision'}, 'Missing unresolved decision')
        _require(row['text'] is None and row['decision'] == 'unresolved', 'Unresolved candidate supplies a target')
    source_maps = {cohort: _source_map(labels_v3, cohort) for cohort in COHORTS_V2}
    _require(not set(source_maps['inherited32']).intersection(source_maps['new58']), 'Inherited/new source identity overlap')
    annotations = _index(labels_v3['inherited_annotations'], 'inherited annotations')
    annotation_links = []
    for row in annotations.values():
        _fields(row, {'source_id', 'sample_ids', 'transcription_lines', 'printed_components'}, 'Incomplete inherited annotation')
        _require(isinstance(row['sample_ids'], list) and isinstance(row['transcription_lines'], list)
                 and 1 <= len(row['sample_ids']) == len(row['transcription_lines']) <= 2, 'Invalid inherited annotation lines')
        annotation_links.extend(row['sample_ids'])
    _require(len(annotation_links) == len(set(annotation_links)) and set(annotation_links) == set(inherited), 'Inherited annotation coverage differs')
    metadata = []
    for cohort, samples in [('inherited32', inherited), ('new58', new)]:
        used_sources = set()
        for identifier, sample in samples.items():
            _fields(sample, {'source_id', 'text', 'image_pin'}, 'Incomplete source sample binding')
            source_id = _identity(sample['source_id'])
            _require(source_id in source_maps[cohort], 'Unknown sample source')
            used_sources.add(source_id)
            record = records[identifier]
            _fields(record, {'source_id', 'source_sha256', 'image_sha256', 'text', 'task', 'split'}, 'Incomplete train record binding')
            _require(record['task'] == 'ocr_recognition' and record['split'] == 'train', 'Only complete OCR train records are supported')
            _require(record['source_id'] == source_id and record['source_sha256'] == source_maps[cohort][source_id]['sha256']
                     and canonical_sha(record['image_sha256']) == _pin(sample['image_pin'])['sha256']
                     and isinstance(sample['text'], str) and sample['text'].strip() and record['text'] == sample['text'],
                     'Train text/source/image revision differs from labels')
            if cohort == 'inherited32':
                _fields(sample, {'annotation_id', 'line_index', 'review_status'}, 'Missing inherited line membership')
                _require(sample['review_status'] == 'accepted_for_ocr_transcription', 'Inherited line is not accepted text')
                index = sample['line_index']; annotation = annotations.get(sample['annotation_id'])
                _require(annotation is not None and type(index) is int and 0 <= index < len(annotation['sample_ids']), 'Invalid inherited line index')
                _require(annotation['source_id'] == source_id and annotation['sample_ids'][index] == identifier
                         and annotation['transcription_lines'][index] == sample['text'], 'Inherited annotation membership differs')
                components = annotation['printed_components']
                _fields(components, {'notation_kind'}, 'Missing inherited notation kind')
                _require(components['notation_kind'] in INHERITED_KINDS_V2, 'Unknown inherited notation kind')
                role = 'inherited_quantity' if index == 1 else 'inherited_'+components['notation_kind']
                role_group = 'quantity' if index == 1 else 'dimension'
            else:
                _fields(sample, {'role', 'decision'}, 'Missing new role or decision')
                _require(sample['decision'] == 'accepted_for_ocr_transcription' and sample['role'] in ROLE_GROUPS_V2, 'Unknown or unaccepted new role')
                role, role_group = sample['role'], ROLE_GROUPS_V2[sample['role']]
            metadata.append({'id': identifier, 'cohort': cohort, 'source_id': source_id, 'role': role, 'role_group': role_group})
        _require(used_sources == set(source_maps[cohort]), 'Source metadata contains an unused or missing source')
    return sorted(metadata, key=lambda row: row['id'])


def _score(rows):
    return aggregate_scores_v1([{key: row[key] for key in SCORE_FIELDS} for row in rows])


def _compact(scores):
    return {key: deepcopy(scores[key]) for key in ('sample_count', 'status_counts', 'raw', 'normalized')}


def _stable(value):
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Prediction evidence must be finite JSON data') from error


def _predictions(rows, name):
    result = _index(rows, name)
    for row in result.values():
        _require(set(row) == PREDICTION_FIELDS, 'Unexpected prediction fields')
        canonical_sha(row['image_sha256']); relative_path(row['image'])
        _require(isinstance(row['detections'], list) and isinstance(row['order'], list), 'Missing detection evidence')
        _require(all(type(index) is int for index in row['order']) and sorted(row['order']) == list(range(len(row['detections']))), 'Invalid detection order permutation')
        if row['status'] == 'error':
            _require(isinstance(row['error'], dict) and set(row['error']) == {'stage', 'type', 'message'}
                     and row['error']['stage'] in ('decode', 'recognize', 'combine')
                     and all(isinstance(value, str) for value in row['error'].values()), 'Invalid error evidence')
        else:
            _require(row['error'] is None, 'Non-error prediction has error evidence')
        _stable(row)
    _score(rows)  # Preserve V1's denominator/status/reference validation.
    return result


def score_strata_v2(predictions, metadata, previous_predictions):
    """Score all strata and report exact inherited-output differences.

    Caller authenticates metadata and current references against its frozen
    inventory. Previous source IDs, references and image hashes must match the
    inherited subset. A changed OCR result is reported rather than hidden or
    rejected; it is a regression observation, not permission to retune a model.
    """
    strata = _index(metadata, 'stratum metadata')
    for row in strata.values():
        _require(set(row) == METADATA_FIELDS and row['cohort'] in COHORTS_V2, 'Unexpected stratum fields or cohort')
        _identity(row['source_id'])
        if row['cohort'] == 'inherited32':
            allowed = {'inherited_'+kind: 'dimension' for kind in INHERITED_KINDS_V2} | {'inherited_quantity': 'quantity'}
        else:
            allowed = ROLE_GROUPS_V2
        _require(row['role'] in allowed and row['role_group'] == allowed[row['role']], 'Stratum role group differs from fixed mapping')
    current = _predictions(predictions, 'current predictions')
    previous = _predictions(previous_predictions, 'previous predictions')
    _require(set(current) == set(strata), 'Predictions do not cover the complete stratum inventory')
    _require(all(row['source_id'] == strata[identifier]['source_id'] for identifier, row in current.items()), 'Prediction source differs from stratum metadata')
    inherited_ids = {identifier for identifier, row in strata.items() if row['cohort'] == 'inherited32'}
    _require(set(previous) == inherited_ids and inherited_ids and inherited_ids != set(strata), 'Previous registry must cover exactly the inherited subset')
    changes = []
    for identifier in sorted(inherited_ids):
        before, after = previous[identifier], current[identifier]
        _require(all(before[key] == after[key] for key in ('source_id', 'reference', 'image_sha256')), 'Inherited source/reference/image changed')
        changed = [key for key in REGRESSION_FIELDS_V2 if _stable(before[key]) != _stable(after[key])]
        if changed:
            changes.append({'id': identifier, 'changed_fields': changed,
                            'previous': {key: deepcopy(before[key]) for key in changed},
                            'current': {key: deepcopy(after[key]) for key in changed}})
    overall = _score(predictions)
    result = {'schema_version': 'ocr-stratified-scores/2.0', 'overall': overall}
    for field in ('cohort', 'role_group', 'role'):
        result['by_'+field] = {value: _compact(_score([current[identifier] for identifier in sorted(strata) if strata[identifier][field] == value]))
                              for value in sorted({row[field] for row in strata.values()})}
    before = _compact(_score(list(previous.values())))
    after = _compact(_score([current[identifier] for identifier in sorted(inherited_ids)]))
    regression = {'sample_count': len(inherited_ids), 'compared_fields': list(REGRESSION_FIELDS_V2),
        'unchanged_count': len(inherited_ids)-len(changes), 'changed_count': len(changes),
        'changed_ids': [row['id'] for row in changes], 'changes': changes, 'previous': before, 'current': after,
        'scope': 'same_inherited_lines_not_independent_evaluation'}
    for mode in ('raw', 'normalized'):
        regression[mode+'_delta'] = {key: after[mode][key]-before[mode][key]
                                    for key in ('exact_matches', 'distance', 'reference_characters', 'cer')}
    result['regression'] = regression
    return result
