"""Independent, read-only verification of versioned reference remediation including pre-callback skips.

Frozen source preparation and comparison evidence remain pinned. Missing native
resources invalidate effective visual verification even if raw overlap passes.
"""
from collections import Counter
import math
from pathlib import Path
import struct
from urllib.parse import unquote, urlsplit
import zlib

import ezdxf
import numpy as np

from .clear_pairs import contained, digest, project_dimension_evidence, read_json
from .pair_update_validation import _Links, _index, _matrix, _normalization, _pin, _png, _require

REQUIRED_DEPENDENCIES = {'intake', 'prepared_batch', 'prepared_validation', 'parent_report',
    'parent_summary', 'parent_validation', 'parent_handoff', 'parent_final_verification',
    'registration_queue', 'full_intake', 'previous_pair_report', 'family_catalog', 'old_split',
    'semantics_recovery_v1', 'remediation_v2_report', 'remediation_v2_summary',
    'remediation_v2_validation', 'remediation_v2_config', 'remediation_v2_resource_audit'}


def _preserved_parent_pixels(parent, pair, old_run, run):
    for kind in ('source', 'target', 'overlay'):
        _require((kind in parent['artifacts']) == (kind in pair['artifacts']), 'Parent pixel artifact inventory changed')
        if kind in parent['artifacts']:
            old_path = contained(old_run, parent['artifacts'][kind])
            _pin(old_path, parent['artifact_sha256'][kind])
            _require(digest(contained(run, pair['artifacts'][kind])) == digest(old_path),
                     'Previously successful review pixels changed: '+kind)
    _require(pair.get('diagnostic_evidence', pair['evidence']) == parent['evidence'],
             'Previously successful raw comparison evidence changed')
    for key in ('source_provenance', 'source_normalization'):
        _require(pair[key] == parent[key], 'Previously successful source preparation changed')
    if pair['reference_complete']:
        _require(pair['alignment'] == parent['alignment'] and pair['evidence'] == parent['evidence']
                 and pair['visual_status'] == parent['visual_status'] and pair['status'] == parent['status'],
                 'Previously complete comparison changed')


def _preserved_v2_result(parent, pair, old_run, run):
    """V3 changes completeness evidence, never the raster or numeric result."""
    for kind in ('source', 'target', 'overlay', 'semantics'):
        _require((kind in parent['artifacts']) == (kind in pair['artifacts']), 'V2 artifact inventory changed')
        if kind in parent['artifacts']:
            old_path = contained(old_run, parent['artifacts'][kind])
            _pin(old_path, parent['artifact_sha256'][kind])
            _require(digest(contained(run, pair['artifacts'][kind])) == digest(old_path),
                     'V3 changed sealed V2 pixels or native semantics: '+kind)
    _require(pair.get('diagnostic_evidence', pair['evidence'])
             == parent.get('diagnostic_evidence', parent['evidence']), 'V3 changed sealed V2 raw comparison evidence')
    _require(pair['source_provenance'] == parent['source_provenance']
             and pair['source_normalization'] == parent['source_normalization'], 'V3 source preparation differs from V2')


def _native_resource_instances(document):
    """Visit source entities before copying, including noncopyable OLE entities.

    Modelspace and INSERT reachability are read directly from native blocks.
    Layer 0 inherits the nearest INSERT layer; INSERT layer state alone does
    not suppress explicitly layered children, matching the pinned frontend.
    No virtual-entity iterator is used, so copy failures cannot erase evidence.
    """
    instances = []
    visited_count = 0

    def visit(layout, insert_path=(), inherited_layer=None, active_blocks=()):
        nonlocal visited_count
        _require(len(active_blocks) <= 32, 'Native resource traversal exceeds nesting budget')
        for entity in layout:
            visited_count += 1
            _require(visited_count <= 1000000, 'Native resource traversal exceeds entity budget')
            kind = entity.dxftype()
            if kind not in ('INSERT', 'IMAGE', 'OLE2FRAME', 'DIMENSION'):
                continue
            if entity.dxf.get('invisible', 0):
                continue
            layer = entity.dxf.get('layer', '0')
            if layer == '0' and inherited_layer is not None:
                layer = inherited_layer
            if kind == 'INSERT':
                block_name = entity.dxf.name
                _require(block_name not in active_blocks and block_name in document.blocks,
                         'Cyclic or missing block in native resource traversal')
                count = entity.mcount
                _require(type(count) is int and 0 < count <= 4096, 'Resource INSERT array exceeds budget')
                for _ in range(count):
                    visit(document.blocks[block_name], (*insert_path, entity.dxf.handle), layer,
                          (*active_blocks, block_name))
                continue
            if layer in document.layers:
                table = document.layers.get(layer)
                if table.is_off() or table.is_frozen():
                    continue
            if kind == 'DIMENSION':
                block_name = entity.dxf.get('geometry')
                if block_name and block_name in document.blocks:
                    # Dimensions use virtual entities too; source blocks must
                    # be inspected before any unsupported entity is dropped.
                    _require(block_name not in active_blocks, 'Cyclic dimension block')
                    visit(document.blocks[block_name], insert_path, inherited_layer,
                          (*active_blocks, block_name))
                continue
            _require(len(instances) < 4096, 'Native resource inventory exceeds budget')
            instances.append({'entity_type': kind, 'native_handle': entity.dxf.handle,
                              'native_owner_handle': entity.dxf.owner, 'insert_path': list(insert_path),
                              'effective_layer': layer})

    visit(document.modelspace())
    return instances


def _visible_resource_instances(document, source, dataset):
    """Inventory callback-reachable entities, not a completeness proof.

    This separate traversal ignores supplied observation lists and raster data.
    The pinned ezdxf visibility rules handle off layers and unused blocks.
    """
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.recorder import Recorder
    from .reference_resources import ResourceAwareFrontendMixin

    class InventoryFrontend(ResourceAwareFrontendMixin, Frontend):
        def __init__(self, *args, **kwargs):
            self.skipped_resources = []
            super().__init__(*args, **kwargs)

        def draw_entity(self, entity, properties):
            if entity.dxftype() in ('IMAGE', 'OLE2FRAME'):
                identity = self._resource_identity(entity)
                self.resource_observations.append(identity)
            elif hasattr(entity, '__virtual_entities__'):
                super().draw_entity(entity, properties)

        def skip_entity(self, entity, msg):
            if entity.dxftype() not in ('IMAGE', 'OLE2FRAME'):
                return
            original = entity.origin_of_copy or entity
            properties = self.ctx.resolve_all(entity)
            self.skipped_resources.append({'entity_type': original.dxftype(),
                'entity_handle': entity.dxf.get('handle'),
                'native_handle': original.dxf.get('handle'), 'native_owner_handle': original.dxf.get('owner'),
                'insert_path': None, 'source_sha256': self._resource_source_sha,
                'native_skip_reason': str(msg), 'resource_required': bool(properties.is_visible),
                'resolved_layer': properties.layer, 'provenance_status': 'native_handle_and_owner_from_frontend_skip',
                'visibility': 'visible' if properties.is_visible else 'hidden'})

    frontend = InventoryFrontend(RenderContext(document), Recorder())
    frontend.configure_resources(source, dataset, True)
    frontend.draw_layout(document.modelspace(), finalize=False)
    return (Counter((row['entity_type'], row['native_handle'], tuple(row['insert_path']))
                    for row in frontend.resource_observations), frontend.skipped_resources)


def _reference_policy(source, reference, pair, dataset):
    """Check source-native identity and conservative effective evidence separately."""
    from .mtext_zero_height import install_zero_height_adapter

    _require(reference.get('method') == 'ezdxf-recorder-explicit-placement-pymupdf-v5-complete-resource-skip-audit',
             'Reference renderer version differs')
    observations = reference.get('resource_observations')
    _require(isinstance(observations, list) and len(observations) <= 4096, 'Invalid resource inventory')
    incomplete = any(row.get('status') == 'incomplete_reference' for row in observations)
    complete = reference.get('reference_complete')
    _require(type(complete) is bool and complete is not incomplete and pair.get('reference_complete') is complete,
             'Reference completeness disagrees with missing resources')
    _require(pair.get('reference_status') == ('complete' if complete else 'incomplete_diagnostic'),
             'Reference status differs from completeness')
    _require(reference.get('reference_usage') == ('comparison_candidate' if complete else 'diagnostic_only_incomplete_reference')
             and reference.get('incomplete_reference_explicitly_allowed') is True,
             'Diagnostic reference policy differs')
    evidence = pair.get('evidence', {})
    if complete:
        _require('diagnostic_evidence' not in pair and not evidence.get('diagnostic_only')
                 and pair['raw_visual_status'] == pair['visual_status'], 'Complete reference has inconsistent raw status')
    else:
        raw = pair.get('diagnostic_evidence')
        _require(isinstance(raw, dict) and pair['raw_visual_status'] == raw.get('status')
                 and pair['status'] == pair['visual_status'] == evidence.get('status') == 'quarantine'
                 and evidence == {**raw, 'status': 'quarantine', 'diagnostic_only': True,
                                   'reasons': [*raw.get('reasons', []), 'incomplete_reference_resources']}
                 and (pair.get('alignment') is None or pair['alignment']['status'] == 'diagnostic_only'),
                 'Incomplete reference escaped effective quarantine')
    doc = ezdxf.readfile(source)
    doc.audit()
    expected = install_zero_height_adapter(doc)
    actual = reference.get('zero_height_mtext_policy')
    _require(isinstance(actual, dict), 'Missing zero-height MTEXT provenance')
    for key, value in expected.items():
        if key not in ('translations', 'translated_instances'):
            _require(actual.get(key) == value, 'Native zero-height proof differs: '+key)
    events = actual.get('translations')
    _require(isinstance(events, list) and len(events) <= 100000
             and actual.get('translated_instances') == len(events), 'Translation event count differs')
    native_handles = {row['source_handle'] for row in expected['native_entities']}
    possible_inputs = {handle: [list(doc.entitydb[handle].dxf.insert)] for handle in native_handles}
    for event in events:
        handle = event.get('source_handle')
        _require(handle in native_handles, 'Translation event references a non-native zero-height MTEXT')
        before = np.asarray(event.get('input_insert_xyz'), dtype=float)
        after = np.asarray(event.get('output_insert_xyz'), dtype=float)
        offset = np.asarray(event.get('translation_xyz'), dtype=float)
        _require(before.shape == after.shape == offset.shape == (3,)
                 and np.isfinite([before, after, offset]).all()
                 and np.array_equal(before+offset, after), 'Translation event changes geometry beyond its offset')
        _require(any(np.array_equal(before, known) for known in possible_inputs[handle]),
                 'Translation event does not start at native or previously translated insertion')
        possible_inputs[handle].append(after)
        _require(event.get('char_height_preserved') == 0 and event.get('inline_text_unchanged') is True
                 and event.get('width_unchanged') is True, 'Translation event changed native text attributes')
    skipped_observations, callback_observations = [], []
    for row in observations:
        entity = doc.entitydb.get(row.get('native_handle'))
        skipped = row.get('observation_kind') == 'skipped_before_resource_callback'
        _require(entity is not None and entity.dxftype() == row.get('entity_type')
                 and row['entity_type'] in ('IMAGE', 'OLE2FRAME')
                 and entity.dxf.get('owner') == row.get('native_owner_handle')
                 and row.get('source_sha256') == reference['source_sha256'], 'Resource observation differs from native source')
        if skipped:
            _require(row.get('insert_path') is None and row.get('provenance_status') == 'native_handle_and_owner_from_frontend_skip'
                     and row.get('reason') == 'native_resource_skipped_before_callback'
                     and row.get('native_skip_reason') in ('non copyable', 'non transformable', 'unsupported non-uniform scaling')
                     and row.get('loaded') is False and row.get('resource_required') is True
                     and row.get('diagnostic_placeholder') is False and row.get('reference_complete') is False
                     and row.get('status') == 'incomplete_reference',
                     'Skipped resource must retain explicit incomplete native-handle-only evidence')
            skipped_observations.append(row)
            continue
        callback_observations.append(row)
        path = row.get('insert_path')
        _require(isinstance(path, list) and len(path) <= 32, 'Invalid resource INSERT provenance')
        owner = doc.modelspace().block_record_handle
        for handle in path:
            insert = doc.entitydb.get(handle)
            _require(insert is not None and insert.dxftype() == 'INSERT' and insert.dxf.owner == owner,
                     'Resource INSERT path is not reachable from modelspace')
            owner = doc.blocks[insert.dxf.name].block_record_handle
        _require(entity.dxf.owner == owner, 'Observed resource is outside its declared INSERT path')
        if entity.dxftype() == 'OLE2FRAME':
            _require(row.get('reason') == 'opaque_ole_content_not_rendered_by_native_frontend'
                     and row.get('status') == 'incomplete_reference', 'OLE resource completeness is unsupported')
        else:
            definition = entity.image_def
            _require(definition is not None and row.get('image_def_handle') == entity.dxf.image_def_handle
                     and row.get('declared_filename') == definition.dxf.get('filename', '')
                     and row.get('image_size_px') == list(entity.dxf.image_size)[:2]
                     and row.get('show_image') == bool(int(entity.dxf.flags) & 1),
                     'IMAGE declaration differs from native source')
            # Repeat source-local resolution with a fresh, unmodified context.
            from .reference_resources import ResourceAwareFrontendMixin
            from types import SimpleNamespace
            resolver = ResourceAwareFrontendMixin()
            resolver.configure_resources(source, dataset, True)
            resolver.ctx = SimpleNamespace(document_dir=source.parent)
            resolved, reason = resolver._resolve_declared_image(row['declared_filename'])
            if row['show_image']:
                if reason is not None:
                    _require(row.get('status') == 'incomplete_reference' and row.get('reason') == reason,
                             'Resource failure differs from current source-local resolution')
                elif row.get('status') == 'loaded_pinned_resource':
                    _require(row.get('resource') == resolver._check_image(resolved), 'Loaded resource pin differs')
                else:
                    _require(row.get('status') == 'incomplete_reference'
                             and row.get('reason') in ('image_validation_failed', 'image_policy_does_not_display_native_content'),
                             'Unsupported resource failure evidence')
            else:
                _require(row.get('status') == 'bitmap_not_requested_by_entity_flags'
                         and row.get('reason') == 'show_image_flag_disabled' and row.get('resource_required') is False,
                         'Hidden bitmap incorrectly reported as required')
        if row.get('status') == 'incomplete_reference':
            _require(row.get('loaded') is False and row.get('resource_required') is True
                     and row.get('diagnostic_placeholder') is True and row.get('reference_complete') is False,
                     'Missing content must remain an explicit diagnostic placeholder')
    callback_resources, expected_skips = _visible_resource_instances(doc, source, dataset)
    _require(reference.get('resource_skip_audit_complete') is True
             and reference.get('resource_skip_count') == len(expected_skips)
             and reference.get('resource_skips') == expected_skips,
             'Complete resource skip audit differs from independent uncapped capture')
    _require(reference.get('callback_reference_complete') is not any(
                row.get('status') == 'incomplete_reference' for row in callback_observations),
             'Callback completeness differs from its own resource observations')
    expected_gaps = [{**row, 'observation_kind': 'skipped_before_resource_callback',
                      'reason': 'native_resource_skipped_before_callback', 'status': 'incomplete_reference',
                      'reference_complete': False, 'loaded': False, 'diagnostic_placeholder': False}
                     for row in expected_skips if row['resource_required']]
    _require(skipped_observations == reference.get('resource_gaps') == expected_gaps,
             'Missing-resource gaps differ from complete source skip capture')
    policy = reference.get('resource_completeness_policy')
    _require(isinstance(policy, dict), 'Missing full resource-completeness policy')
    hidden = sum(not row['resource_required'] for row in expected_skips)
    expected_policy = {'method': 'drawn-and-skipped-native-resource-completeness-v1',
        'full_resource_skip_audit': True,
        'generic_skip_log_truncated': reference['skipped_entity_count'] > len(reference['skipped_entities']),
        'callback_observation_count': len(callback_observations), 'captured_resource_skip_count': len(expected_skips),
        'hidden_resource_skip_count': hidden, 'required_resource_skip_count': len(expected_skips)-hidden,
        'unknown_resource_audit_gap_count': 0, 'max_resource_skip_records': 4096,
        'insert_path_policy': 'Skip callbacks do not establish the INSERT path; null is not an empty direct-modelspace path.'}
    _require(policy == expected_policy, 'Resource-completeness policy counters or scope differ')
    _require(callback_resources == Counter((row['entity_type'], row['native_handle'], tuple(row['insert_path']))
                                          for row in callback_observations),
             'Resource callback inventory differs from independent dispatch walk')
    native_instances = _native_resource_instances(doc)
    native_resources = Counter((row['entity_type'], row['native_handle'], tuple(row['insert_path']))
                               for row in native_instances)
    _require(not callback_resources-native_resources, 'Callback resource was not reached in original native blocks')
    missing = native_resources-callback_resources
    missing_handles = Counter()
    for (kind, handle, _), count in missing.items():
        missing_handles[(kind, handle)] += count
    _require(missing_handles == Counter((row['entity_type'], row['native_handle']) for row in skipped_observations),
             'Native visible resource omitted before callback without complete skip evidence')
    _require(complete is not bool(missing or incomplete), 'Native resource gap escaped completeness quarantine')
    _pin(source, reference['source_sha256'])
    return {'native_visible_resource_count': sum(native_resources.values()),
            'callback_visible_resource_count': sum(callback_resources.values()),
            'skipped_visible_resource_count': sum(missing.values()),
            'skipped_native_instances': [row for row in native_instances
                if (row['entity_type'], row['native_handle'], tuple(row['insert_path'])) in missing],
            'proof_method': 'native-modelspace-insert-traversal-before-copy-v1',
            'source_sha256': reference['source_sha256']}


def validate_pair_update_v3(workspace: Path, config_path: Path, run: Path, summary: Path) -> dict:
    root = Path(workspace).resolve()
    config_path, run, summary = Path(config_path).resolve(), Path(run).resolve(), Path(summary).resolve()
    _require(all(p.is_relative_to(root) for p in (config_path, run, summary)), 'Review inputs must stay in workspace')
    config_sha = digest(config_path)
    config = read_json(config_path)
    _require(config.get('schema_version') == '1.0' and config.get('purpose') == 'versioned_reference_remediation_pair_review',
             'Expected registered pair update configuration')
    _require(config.get('training_eligible') is False and config.get('split_modified') is False
             and config.get('source_modified') is False, 'Configuration must preserve sources and unapproved training state')
    dependencies = config['dependencies']
    _require(REQUIRED_DEPENDENCIES <= set(dependencies), 'Missing review dependency')
    paths, data = {}, {}
    for name, record in dependencies.items():
        paths[name] = contained(root, record['path'])
        _pin(paths[name], record['sha256'])
        data[name] = read_json(paths[name])
    for relative, expected in config['entrypoints'].items():
        _pin(contained(root, relative), expected)
    for record in config['library_dependencies'].values():
        _pin(contained(root, record['path']), record['sha256'])
    dataset = contained(root, config['dataset'])
    _require(not run.is_relative_to(dataset) and not summary.is_relative_to(dataset), 'Review artifacts must be outside dataset')
    report_path, summary_path = run/'report.json', summary/'summary.json'
    report_sha, summary_sha = digest(report_path), digest(summary_path)
    report, reviewed = read_json(report_path), read_json(summary_path)
    count = config['expected_count']
    _require(type(count) is int and 0 < count <= 10000, 'Invalid expected source count')
    _require(report.get('status') == 'complete' and report.get('complete') is True
             and report.get('candidate_count') == report.get('processed_count') == count, 'Expected complete processed pair review')
    _require(report.get('training_eligible') is False and report.get('split_modified') is False,
             'Pair review must not approve training or change splits')
    _require(report['intake_sha256'] == dependencies['intake']['sha256']
             and report['preparation_batch_sha256'] == dependencies['prepared_batch']['sha256'], 'Review dependencies differ')
    old = data['parent_report']
    _require(data['parent_summary']['run_report_sha256'] == dependencies['parent_report']['sha256'],
             'Parent summary belongs to another run')
    _require(data['parent_validation']['status'] == 'verified'
             and data['parent_validation']['run_report_sha256'] == dependencies['parent_report']['sha256'],
             'Parent validation belongs to another run')
    for key in ('comparison_method', 'thresholds', 'max_side'):
        _require(report[key] == old[key] == config[key], 'Pair comparison changed from frozen parent: '+key)
    _require(report['implementation'] == config['implementation'], 'Versioned implementation differs from configuration')
    for name, expected in report['implementation'].items():
        _pin(contained(Path(__file__).parent, name), expected)
    intake, batch, proven = data['intake'], data['prepared_batch'], data['prepared_validation']
    _require(batch.get('status') == 'complete' and batch['intake_sha256'] == dependencies['intake']['sha256'], 'Prepared batch differs from intake')
    _require(proven.get('status') == 'verified' and proven['batch_report_sha256'] == dependencies['prepared_batch']['sha256']
             and proven['intake_sha256'] == dependencies['intake']['sha256'], 'Preparation proof differs from batch')
    items, prepared, validated = _index(intake['items']), _index(batch['items']), _index(proven['pages'])
    pairs, rows = _index(report['pairs']), _index(reviewed['review_queue'])
    full, queue = _index(data['full_intake']['items']), _index(data['registration_queue']['items'])
    _require(len(items) == count and set(items) == set(prepared) == set(validated) == set(pairs) == set(rows), 'New review inventory differs')
    _require(set(full) == set(queue) and set(items) <= set(full), 'Combined registered inventory differs')
    _require(reviewed.get('status') == 'passed' and reviewed['run_report_sha256'] == report_sha
             and reviewed.get('training_eligible') is False, 'Artifact summary is not for this complete run')
    v2_report = data['remediation_v2_report']
    v2_pairs = _index(v2_report['pairs'])
    _require(v2_report.get('status') == 'complete' and v2_report.get('complete') is True
             and set(v2_pairs) == set(items), 'V2 review inventory or completion differs')
    _require(data['remediation_v2_summary']['run_report_sha256'] == dependencies['remediation_v2_report']['sha256']
             and data['remediation_v2_validation']['run_report_sha256'] == dependencies['remediation_v2_report']['sha256']
             and data['remediation_v2_validation']['config_sha256'] == dependencies['remediation_v2_config']['sha256'],
             'V2 sealed run, summary, validation or config provenance differs')
    for name, expected in data['remediation_v2_config']['implementation'].items():
        _pin(contained(Path(__file__).parent, name), expected)
    parent_pairs = _index(old['pairs'])
    _require(set(parent_pairs) == set(items), 'Versioned review differs from parent inventory')
    recovery_path = paths['semantics_recovery_v1']
    recovered = data['semantics_recovery_v1']
    _require(recovered.get('status') == 'recovered_serialization_only'
             and recovered.get('run_report_sha256') == dependencies['parent_report']['sha256']
             and recovered.get('training_eligible') is False and recovered.get('source_modified') is False,
             'Recovery index must preserve parent sources and semantics')
    recovery = {item['input_path']: item for item in recovered['items']}
    _require(len(recovery) == len(recovered['items']), 'Repeated recovery identity')
    needed = {name for name, pair in parent_pairs.items() if 'semantics' not in pair['artifacts']
              and any(e.get('stage') == 'semantics' and e.get('type') == 'UnicodeEncodeError' for e in pair['errors'])}
    _require(set(recovery) == needed, 'Recovery inventory differs from exact serialization failures')
    old_inputs = {pair['input_path'] for pair in data['previous_pair_report']['pairs']}
    _require(not old_inputs & set(items) and old_inputs | set(items) == set(full), 'Old and new review identities overlap or leave gaps')
    prepared_root = paths['prepared_batch'].parent
    entries, visual, pair_status, units, display = [], Counter(), Counter(), Counter(), Counter()
    artifacts_seen, sources_seen, target_names = set(), set(), set()
    totals = Counter()
    resources, raw_visual, reference_status = Counter(), Counter(), Counter()
    policy_changes = []
    for ordinal, item in enumerate(intake['items'], 1):
        name = item['input_path']
        pair, prep_item, proof, row = pairs[name], prepared[name], validated[name], rows[name]
        local_id = f'pair-{ordinal:04d}'
        pair_file = run/local_id/'pair.json'
        _require(read_json(pair_file) == pair, 'Local pair report differs from batch record')
        errors = pair.get('errors')
        _require(errors == [], 'Remediated review contains execution errors')
        _require(isinstance(errors, list) and pair.get('training_eligible') is False
                 and pair.get('annotation_status') == 'not_approved', 'Pair errors or unapproved labels changed')
        if errors:
            if any(e.get('stage') == 'pair' for e in errors):
                _require(pair['status'] == pair['visual_status'] == 'quarantine' and pair.get('alignment') is None,
                         'Pair execution failures must remain unregistered quarantine')
            _require(all(isinstance(e, dict) and e.get('stage') in ('pair', 'semantics')
                         and isinstance(e.get('type'), str) and isinstance(e.get('message'), str) for e in errors),
                     'Malformed execution error evidence')
            totals['pairs_with_execution_errors'] += 1
        _require(item.get('training_eligible') is False and item.get('split') is None
                 and full[name]['input_sha256'] == item['input_sha256'], 'New intake identity or training state differs')
        target = item['candidate_target']
        _require(target['path'] not in target_names, 'One target is reused by multiple inputs')
        target_names.add(target['path'])
        for side, path_name, expected in [('input', name, item['input_sha256']), ('target', target['path'], target['sha256'])]:
            _require(pair[side+'_path'] == path_name and pair[side+'_sha256'] == expected, 'Pair candidate differs from registered source')
            source = contained(dataset, path_name)
            _pin(source, expected)
            sources_seen.add(source)
        source_path = contained(dataset, name)
        _require(source_path.stat().st_size == item['input_size_bytes'], 'Original source size differs')
        with source_path.open('rb') as stream:
            header = stream.read(33)
        _require(header[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
                 and zlib.crc32(header[12:29]) & 0xffffffff == struct.unpack('>I', header[29:33])[0], 'Original PNG header differs')
        dimensions = list(struct.unpack('>II', header[16:24]))
        _require(dimensions == [item['png_ihdr']['width_px'], item['png_ihdr']['height_px']], 'Original PNG dimensions differ')
        prep_path = contained(prepared_root, prep_item['preparation_report'])
        _pin(prep_path, prep_item['preparation_report_sha256'])
        page = read_json(prep_path)
        provenance = pair.get('source_provenance')
        _require(isinstance(provenance, dict) or errors, 'Missing prepared source provenance')
        _require(page['source_sha256'] == item['input_sha256'] == proof['source_sha256']
                 and prep_item['preparation_report_sha256'] == proof['preparation_report_sha256']
                 and page['source_size_px'] == dimensions,
                 'Prepared page provenance differs from pinned pixel validation')
        overview = page['overview']
        overview_path = contained(prep_path.parent, overview['path'])
        _pin(overview_path, overview['sha256'])
        if provenance is not None:
            _require(provenance['overview_sha256'] == overview['sha256']
                     and provenance['preparation_report'] == prep_item['preparation_report']
                     and provenance['preparation_report_sha256'] == prep_item['preparation_report_sha256']
                     and provenance['source_size_px'] == dimensions, 'Review overview identity differs')
        factor = overview['scale_factor']
        overview_matrix = _matrix(overview['overview_to_source'], [[factor, 0, (factor-1)/2], [0, factor, (factor-1)/2], [0, 0, 1]])
        if provenance is not None:
            _matrix(provenance['overview_to_source'], overview_matrix)
        gray = _png(overview_path, expected_size=overview['size_px'], grayscale=True)
        normalization = pair.get('source_normalization')
        _require(isinstance(normalization, dict) or errors, 'Missing source normalization')
        expected_pixels, normalized_inverse = (None, None) if normalization is None else _normalization(gray, normalization, report['max_side'])
        _require(set(pair['artifacts']) == set(pair['artifact_sha256'])
                 and (errors or {'source', 'target', 'semantics', 'reference'} <= set(pair['artifacts'])), 'Missing or unpinned pair artifacts')
        artifact_paths = {}
        for kind, relative in pair['artifacts'].items():
            path = contained(run, relative)
            _require(path.parent == run/local_id and path not in artifacts_seen, 'Artifact belongs to another pair or repeats')
            artifacts_seen.add(path)
            artifact_paths[kind] = path
            _pin(path, pair['artifact_sha256'][kind])
            if path.suffix.lower() == '.png':
                _png(path)
            totals['checked_artifacts'] += 1
        if 'source' in artifact_paths:
            _require(expected_pixels is not None, 'Source artifact lacks recorded normalization')
            observed_pixels = _png(artifact_paths['source'], expected_size=list(expected_pixels.shape[::-1]), grayscale=True)
            _require(np.array_equal(expected_pixels, observed_pixels), 'Normalized review source differs from pinned overview pixels')
            del observed_pixels
        del gray, expected_pixels
        native_semantic = 'semantics' in artifact_paths
        semantics = read_json(artifact_paths['semantics'])
        parent_pair = parent_pairs[name]
        _preserved_v2_result(v2_pairs[name], pair, paths['remediation_v2_report'].parent, run)
        totals['v2_pixel_and_raw_evidence_results_preserved'] += 1
        if name in recovery:
            record = recovery[name]
            _require(record['input_sha256'] == item['input_sha256']
                     and record['target_path'] == target['path'] and record['target_sha256'] == target['sha256']
                     and record['pair_id'] == local_id and record['method'] == 'json_ensure_ascii_true'
                     and record['native_string_code_points_preserved'] is True and record['source_text_corrected'] is False,
                     'Recovered semantics identity or text changed')
            parent_semantic_path = contained(recovery_path.parent, record['semantics']['path'])
            _pin(parent_semantic_path, record['semantics']['sha256'])
            totals['unicode_serialization_failures_resolved'] += 1
        else:
            parent_semantic_path = contained(paths['parent_report'].parent, parent_pair['artifacts']['semantics'])
            _pin(parent_semantic_path, parent_pair['artifact_sha256']['semantics'])
        _require(semantics == read_json(parent_semantic_path), 'Native semantic records differ from frozen parent or lossless recovery')
        totals['unchanged_semantic_reports'] += 1
        reference = read_json(artifact_paths['reference']) if 'reference' in artifact_paths else None
        if semantics is not None:
            _require(semantics['source_sha256'] == target['sha256'] and semantics.get('training_eligible') is False,
                     'CAD semantics identity or approval differs')
            _require(semantics['inventory_complete'] is True, 'Semantic inventory is incomplete')
        if reference is not None:
            _require(reference['source_sha256'] == target['sha256'] and reference.get('source_modified') is False
                     and reference.get('physical_scale_confirmed') is False and reference.get('cad_semantics_verified') is False,
                     'CAD reference identity or approval differs')
            _matrix(reference['pixel_to_cad'], np.linalg.inv(_matrix(reference['cad_to_pixel'], reference['cad_to_pixel'])))
            totals['reference_reports'] += 1
            native_resource_proof = _reference_policy(contained(dataset, target['path']), reference, pair, dataset)
            totals['native_visible_resource_instances'] += native_resource_proof['native_visible_resource_count']
            totals['pre_callback_resource_skip_instances'] += native_resource_proof['skipped_visible_resource_count']
            if native_resource_proof['skipped_visible_resource_count']:
                totals['sources_with_pre_callback_resource_skips'] += 1
            resources.update(row['reason'] or 'loaded' for row in reference['resource_observations'])
            reference_status[pair['reference_status']] += 1
            raw_visual[pair['raw_visual_status']] += 1
            totals['zero_height_native_entities'] += reference['zero_height_mtext_policy']['native_zero_height_count']
            totals['zero_height_translated_instances'] += reference['zero_height_mtext_policy']['translated_instances']
            if reference['zero_height_mtext_policy']['native_zero_height_count']:
                totals['sources_with_zero_height_adapter'] += 1
            if not reference['reference_complete']:
                totals['incomplete_diagnostic_references'] += 1
            else:
                totals['complete_references'] += 1
            if any(e.get('stage') == 'pair' for e in parent_pair['errors']):
                totals['parent_render_failures_resolved'] += 1
            if not parent_pair['errors']:
                _preserved_parent_pixels(parent_pair, pair, paths['parent_report'].parent, run)
                totals['parent_error_free_results_preserved'] += 1
                if not reference['reference_complete']:
                    policy_changes.append({'input_path': name, 'parent_visual_status': parent_pair['visual_status'],
                                           'visual_status': pair['visual_status'], 'reason': 'incomplete_reference_resources'})
        comparison_available = reference is not None and 'target' in artifact_paths and 'evidence' in pair
        if comparison_available:
            evidence = pair['evidence']
            _require(evidence['method'] == report['comparison_method'] and evidence['status'] == pair['visual_status'], 'Visual status differs from evidence')
            _require(pair['visual_status'] in ('verified', 'quarantine')
                     and pair['status'] == ('review_required' if pair['visual_status'] == 'verified' else 'quarantine'), 'Pair status does not match visual result')
            raw_evidence = pair.get('diagnostic_evidence', evidence)
            if 'metrics' in raw_evidence:
                reasons = ['below_'+key for key, threshold in report['thresholds'].items() if raw_evidence['metrics'][key] < threshold]
                _require(raw_evidence['reasons'] == reasons and (raw_evidence['status'] == 'verified') == (not reasons), 'Visual thresholds or reasons disagree')
            else:
                _require(raw_evidence['status'] == 'quarantine' and bool(raw_evidence['reasons']), 'Unregistered visual match cannot be approved')
            _png(artifact_paths['target'], expected_size=reference['size_px'])
            alignment, affine = pair.get('alignment'), evidence.get('affine_target_to_source')
            if affine is None:
                _require(alignment is None and 'dimension_anchors' not in artifact_paths and 'overlay' not in artifact_paths,
                         'Unregistered pair contains coordinate evidence')
            else:
                _require(isinstance(alignment, dict), 'Registration missing composed coordinates')
                affine = np.asarray(affine, dtype=float)
                _require(affine.shape == (2, 3) and np.isfinite(affine).all(), 'Invalid estimated affine')
                expected = overview_matrix @ normalized_inverse @ np.vstack([affine, [0, 0, 1]]) @ np.asarray(reference['cad_to_pixel'])
                _matrix(alignment['cad_to_source_pixel'], expected)
                _matrix(alignment['source_pixel_to_cad'], np.linalg.inv(expected))
                _require(alignment['annotation_transform_approved'] is False and alignment['physical_scale_confirmed'] is False
                         and alignment['tolerance_is_accuracy_bound'] is False
                         and alignment['status'] == ('visual_alignment_candidate' if pair['visual_status'] == 'verified' else 'diagnostic_only'),
                         'Registration approval state differs')
                tolerance = evidence.get('tolerance_pixels', 0)*np.linalg.norm((overview_matrix @ normalized_inverse)[:2, :2], ord=2)
                _require(math.isclose(alignment['comparison_tolerance_in_source_pixels'], tolerance, rel_tol=1e-12, abs_tol=1e-8), 'Comparison tolerance transform differs')
                _require({'overlay', 'dimension_anchors'} <= set(artifact_paths), 'Registered pair lacks overlay or anchors')
                _require(semantics is not None, 'Serialized semantics recovery is required to verify saved anchors')
                expected_anchors = project_dimension_evidence(semantics, alignment, dimensions)
                expected_anchors['input_sha256'] = item['input_sha256']
                _require(read_json(artifact_paths['dimension_anchors']) == expected_anchors, 'Projected dimension evidence differs')
                _png(artifact_paths['overlay'], expected_size=pair['source_normalization']['normalized_size_px'])
                totals['invertible_registrations'] += 1
        else:
            _require(bool(errors), 'Successful pair lacks comparison evidence')
            alignment = None
            _require('dimension_anchors' not in artifact_paths and 'overlay' not in artifact_paths,
                     'Failed pair retains unverified alignment artifacts')
        repair = page.get('container_repair')
        repair_proof = proof.get('container_repair_provenance')
        if repair is not None:
            _require(page.get('source_png_conformant') is False and isinstance(repair_proof, dict)
                     and repair_proof['source_sha256'] == item['input_sha256']
                     and repair_proof['canonical_png_sha256'] == repair['canonical_png']['sha256']
                     and repair_proof['excluded_trailer_sha256'] == repair['excluded_trailer']['sha256']
                     and repair_proof['original_bytes_equal_canonical_plus_preserved_trailer'] is True,
                     'Repaired original lacks pinned byte-preservation proof')
            partition = []
            for key in ('canonical_png', 'excluded_trailer'):
                artifact = repair[key]
                path = contained(prep_path.parent, artifact['path'])
                _pin(path, artifact['sha256'])
                _require(path.stat().st_size == artifact['size_bytes'], 'Repair partition size changed')
                partition.append(path.read_bytes())
            _require(source_path.read_bytes() == b''.join(partition), 'Original no longer equals preserved repair partition')
            totals['container_repaired_sources'] += 1
        else:
            _require(repair_proof is None and page.get('source_png_conformant') is not False, 'Unexpected repair provenance')
        mismatch = [] if semantics is None else [d for d in semantics['dimensions'] if d['display_comparison']['status'] == 'mismatch']
        _require(row['visual_status'] == pair['visual_status'] and row['target_path'] == target['path']
                 and row['artifacts'] == pair['artifacts'] and row['errors'] == errors
                 and row['reasons'] == pair.get('evidence', {}).get('reasons', [])
                 and row.get('dimensions') == (len(semantics['dimensions']) if native_semantic else None)
                 and len(row.get('numeric_mismatches', [])) == (len(mismatch) if native_semantic else 0)
                 and row['training_eligible'] is False, 'Review summary row differs from pair evidence')
        _require(row.get('reference_complete') is pair['reference_complete']
                 and row.get('reference_status') == pair['reference_status']
                 and row.get('raw_visual_status') == pair['raw_visual_status']
                 and row.get('diagnostic_only') is (not pair['reference_complete']),
                 'Review summary lost reference completeness or diagnostic status')
        registered = queue[name]
        _require(registered['input_sha256'] == item['input_sha256'] and registered.get('training_eligible') is False
                 and registered.get('split') is None
                 and registered['candidate_target']['path'] == target['path']
                 and registered['candidate_target']['sha256'] == target['sha256'], 'Registration queue changed source or approval')
        entries.append({'qualified_pair_id': 'update-v3:'+local_id, 'pair_id': local_id,
            'input_path': name, 'input_sha256': item['input_sha256'], 'target_path': target['path'], 'target_sha256': target['sha256'],
            'pair_report': {'path': pair_file.relative_to(root).as_posix(), 'sha256': digest(pair_file)},
            'preparation': {'path': prep_path.relative_to(root).as_posix(), 'sha256': digest(prep_path)},
            'visual_status': pair['visual_status'], 'pair_status': pair['status'],
            'raw_visual_status': pair['raw_visual_status'], 'reference_status': pair['reference_status'],
            'reference_complete': pair['reference_complete'],
            'reference': {'path': artifact_paths['reference'].relative_to(root).as_posix(), 'sha256': digest(artifact_paths['reference'])},
            'resource_observations': reference['resource_observations'],
            'native_resource_visibility_proof': native_resource_proof,
            'missing_resource_count': sum(r.get('status') == 'incomplete_reference' for r in reference['resource_observations']),
            'adapter_count': reference['zero_height_mtext_policy']['native_zero_height_count'],
            'adapter_translations': reference['zero_height_mtext_policy']['translated_instances'],
            'alignment_status': alignment['status'] if alignment else None,
            'dimension_count': len(semantics['dimensions']) if semantics is not None else None,
            'numeric_mismatch_count': len(mismatch) if semantics is not None else None,
            'units': semantics['units'] if semantics is not None else None,
            'units_name': semantics['units_name'] if semantics is not None else None,
            'visual_reasons': pair.get('evidence', {}).get('reasons', []), 'execution_errors': errors,
            'evidence_completeness': 'complete' if reference['reference_complete'] else 'incomplete_reference_resources',
            'comparison_available': comparison_available,
            'handoff_status': 'technical_quarantine' if not reference['reference_complete'] else pair['status'],
            'raw_semantic_available': native_semantic, 'recovered_semantic_available': name in recovery,
            'semantic_recovery': recovery.get(name),
            'family_review_status': registered['family_review_status'],
            'train_only_constraint_inherited': registered['train_only_constraint_inherited'],
            'source_png_conformant': page.get('source_png_conformant', True), 'container_repair': repair,
            'split': None, 'annotation_approved': False, 'training_eligible': False})
        if semantics is not None:
            totals['reviewed_dimension_instances'] += len(semantics['dimensions'])
            totals['reviewed_numeric_mismatches'] += len(mismatch)
        if native_semantic:
            totals['semantic_reports'] += 1
            totals['dimension_instances'] += len(semantics['dimensions'])
            totals['numeric_mismatches'] += len(mismatch)
        visual[pair['visual_status']] += 1
        pair_status[pair['status']] += 1
        if native_semantic:
            units[semantics['units_name']] += 1
            display.update(d['display_comparison']['status'] for d in semantics['dimensions'])
    _require(dict(reference_status) == report['summary']['reference_status']
             and dict(reference_status) == reviewed['reference_status']
             and dict(raw_visual) == report['summary']['raw_visual_status'] == reviewed['raw_visual_status'],
             'Reference completeness or raw status summary differs')
    _require(report['summary']['errors'] == totals['pairs_with_execution_errors'], 'Execution error count differs')
    _require(dict(visual) == report['summary']['visual_status'] == reviewed['counts']['visual_status']
             and dict(pair_status) == report['summary']['pair_status'], 'Summary visual counts differ')
    _require(dict(units) == reviewed['counts']['units'] and dict(display) == reviewed['counts']['display_comparisons'], 'Summary CAD counts differ')
    for key in ('checked_artifacts', 'semantic_reports', 'dimension_instances', 'invertible_registrations'):
        _require(totals[key] == reviewed['totals'].get(key, 0), 'Summary total differs: '+key)
    _require(reviewed['source_files_rehashed'] == len(sources_seen), 'Summary source count differs')
    probes_meta = reviewed['cross_pair_probes']
    if probes_meta['artifact'] is not None:
        probes = read_json(contained(summary, probes_meta['artifact']))
        _require(len(probes) == probes_meta['count']
                 and dict(Counter(probe['evidence']['status'] for probe in probes)) == probes_meta['visual_status'], 'Cross-probe counts differ')
        for probe in probes:
            _require(probe['input_path'] in pairs and probe['input_sha256'] == pairs[probe['input_path']]['input_sha256']
                     and probe['different_target_path'] in target_names
                     and probe['different_target_path'] != pairs[probe['input_path']]['target_path'], 'Cross probe has invalid source or same target')
    parser = _Links()
    parser.feed((summary/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        parsed = urlsplit(link)
        path = (summary/unquote(parsed.path)).resolve()
        _require(not parsed.scheme and not parsed.netloc and path.is_relative_to(root) and path.is_file(), 'Missing or external summary link')
    for name, record in dependencies.items():
        _pin(paths[name], record['sha256'])
    for relative, expected in config['entrypoints'].items():
        _pin(contained(root, relative), expected)
    for record in config['library_dependencies'].values():
        _pin(contained(root, record['path']), record['sha256'])
    for name, expected in report['implementation'].items():
        _pin(contained(Path(__file__).parent, name), expected)
    for pair in pairs.values():
        for side in ('input', 'target'):
            _pin(contained(dataset, pair[side+'_path']), pair[side+'_sha256'])
    for name, expected in data['remediation_v2_config']['implementation'].items():
        _pin(contained(Path(__file__).parent, name), expected)
    _pin(config_path, config_sha)
    _pin(report_path, report_sha)
    _pin(summary_path, summary_sha)
    return {'schema_version': '1.0', 'status': 'verified', 'config_sha256': config_sha,
        'run_report_sha256': report_sha, 'summary_report_sha256': summary_sha,
        'code_sha256': digest(Path(__file__)), 'dependencies': dependencies,
        'semantics_recovery': dependencies['semantics_recovery_v1'],
        'diagnostic_policy_changes_for_parent_error_free_pairs': policy_changes,
        'summary': {**dict(totals), 'reference_status': dict(reference_status), 'raw_visual_status': dict(raw_visual),
            'resource_reasons': dict(resources), 'pair_count': len(entries), 'artifact_count': totals['checked_artifacts'],
            'source_files_rehashed': len(sources_seen), 'semantic_dimensions': totals['reviewed_dimension_instances'],
            'numeric_mismatches': totals['reviewed_numeric_mismatches'], 'registrations': totals['invertible_registrations'],
            'container_repaired_sources': totals['container_repaired_sources'], 'visual_status': dict(visual),
            'pairs_with_execution_errors': totals['pairs_with_execution_errors'], 'reference_reports': totals['reference_reports'],
            'raw_semantic_reports': totals['semantic_reports'], 'recovered_semantic_reports': totals['unicode_serialization_failures_resolved'],
            'units': dict(units), 'local_html_link_occurrences_verified': len(parser.links)},
        'entries': entries, 'training_eligible': False, 'split_modified': False, 'source_modified': False,
        'limitations': ['No training labels, geometric roles, physical scale or family independence are approved.',
            'The comparison method and thresholds match the previous frozen run.',
            'Composed matrices and anchor evidence are internally consistent; this is not a measurement of registration accuracy.',
            'Source normalization pixels are reproduced from the pinned prepared overview; the original huge RGB is not decoded again.',
            'Native text proof reuses the pinned parser policy; event arithmetic and source handles are checked independently.',
            'Native modelspace/INSERT visibility is audited before copying; callback-only inventories cannot prove completeness of noncopyable resources.',
            'Cross-pair comparisons are diagnostic probes, not labeled independent negatives.']}
