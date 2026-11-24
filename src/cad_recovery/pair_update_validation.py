"""Read-only verification and qualified handoff of a registered pair update.

Preserves the sealed pair pipeline. A successful audit validates recorded
evidence and provenance; it does not approve training labels or CAD geometry.
"""
from collections import Counter
from html.parser import HTMLParser
import math
from pathlib import Path
import struct
from urllib.parse import unquote, urlsplit
import zlib

import cv2
import numpy as np
from PIL import Image

from .clear_pairs import contained, digest, project_dimension_evidence, read_json

REQUIRED_DEPENDENCIES = {'intake', 'prepared_batch', 'prepared_validation',
    'registration_report', 'registration_queue', 'full_intake',
    'previous_pair_report', 'previous_pair_summary', 'family_catalog', 'old_split'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(path, expected):
    _require(isinstance(expected, str) and len(expected) == 64
             and all(c in '0123456789abcdef' for c in expected), 'Invalid SHA-256 pin')
    _require(digest(path) == expected, 'Pinned file changed: '+path.name)


def _index(items, key='input_path'):
    _require(isinstance(items, list) and 0 < len(items) <= 10000, 'Invalid input inventory')
    result = {item[key]: item for item in items}
    _require(len(result) == len(items), 'Repeated inventory identity')
    return result


def _matrix(value, expected):
    actual = np.asarray(value, dtype=float)
    _require(actual.shape == (3, 3) and np.isfinite(actual).all(), 'Invalid coordinate matrix')
    _require(np.allclose(actual, expected, rtol=0, atol=1e-8), 'Coordinate matrix differs from recorded operations')
    return actual


def _png(path, *, expected_size=None, grayscale=False):
    _require(0 < path.stat().st_size <= 50*1024**2, 'Review PNG exceeds byte budget')
    raw = path.read_bytes()
    _require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'Review artifact is not PNG')
    position, count = 8, 0
    while position < len(raw):
        count += 1
        _require(count <= 10000 and position+12 <= len(raw), 'Invalid PNG chunks')
        length = struct.unpack('>I', raw[position:position+4])[0]
        end = position+12+length
        _require(end <= len(raw), 'Truncated PNG chunk')
        _require(zlib.crc32(raw[position+4:end-4]) & 0xffffffff == struct.unpack('>I', raw[end-4:end])[0], 'PNG CRC differs')
        kind = raw[position+4:position+8]
        _require(kind not in (b'acTL', b'fcTL', b'fdAT'), 'Animated review artifact')
        if kind == b'IEND':
            _require(length == 0 and end == len(raw), 'Unexpected PNG trailer')
            break
        position = end
    else:
        raise ValueError('PNG lacks IEND')
    with Image.open(path) as picture:
        size = list(picture.size)
        _require(picture.format == 'PNG' and 0 < picture.width*picture.height <= 16_777_216,
                 'Review image dimensions exceed budget')
        if expected_size is not None:
            _require(size == expected_size, 'Review PNG dimensions differ')
        picture.verify()
    if grayscale:
        with Image.open(path) as picture:
            return np.asarray(picture.convert('L'))
    return size


def _normalization(gray, recorded, max_side):
    ys, xs = np.where(gray < 210)
    _require(len(xs) >= 100, 'Insufficient source overview ink')
    left, top = max(0, int(xs.min())-3), max(0, int(ys.min())-3)
    right, bottom = min(gray.shape[1], int(xs.max())+4), min(gray.shape[0], int(ys.max())+4)
    width, height = right-left, bottom-top
    scale = max_side/max(width, height)
    target_width, target_height = max(1, round(width*scale)), max(1, round(height*scale))
    sx, sy = target_width/width, target_height/height
    forward = np.array([[sx, 0, sx*(.5-left)-.5], [0, sy, sy*(.5-top)-.5], [0, 0, 1.]])
    _require(recorded['input_size_px'] == list(gray.shape[::-1])
             and recorded['crop_box_px'] == [left, top, right, bottom]
             and recorded['normalized_size_px'] == [target_width, target_height], 'Source normalization bounds differ')
    _matrix(recorded['input_to_normalized'], forward)
    inverse = _matrix(recorded['normalized_to_input'], np.linalg.inv(forward))
    pixels = cv2.resize(gray[top:bottom, left:right], (target_width, target_height),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return pixels, inverse


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src') and value)


def validate_pair_update(workspace: Path, config_path: Path, run: Path, summary: Path, *, semantics_recovery_path: Path | None = None) -> dict:
    root = Path(workspace).resolve()
    config_path, run, summary = Path(config_path).resolve(), Path(run).resolve(), Path(summary).resolve()
    _require(all(p.is_relative_to(root) for p in (config_path, run, summary)), 'Review inputs must stay in workspace')
    config_sha = digest(config_path)
    config = read_json(config_path)
    _require(config.get('schema_version') == '1.0' and config.get('purpose') == 'registered_pair_update_review',
             'Expected registered pair update configuration')
    dependencies = config['dependencies']
    _require(REQUIRED_DEPENDENCIES <= set(dependencies), 'Missing review dependency')
    paths, data = {}, {}
    for name, record in dependencies.items():
        paths[name] = contained(root, record['path'])
        _pin(paths[name], record['sha256'])
        data[name] = read_json(paths[name])
    if 'review_entrypoint' in config:
        entrypoint = config['review_entrypoint']
        _pin(contained(root, entrypoint['path']), entrypoint['sha256'])
    dataset = contained(root, config['dataset'])
    _require(not run.is_relative_to(dataset) and not summary.is_relative_to(dataset), 'Review artifacts must be outside dataset')
    report_path, summary_path = run/'report.json', summary/'summary.json'
    report_sha, summary_sha = digest(report_path), digest(summary_path)
    report, reviewed = read_json(report_path), read_json(summary_path)
    count = config['expected_new_count']
    _require(type(count) is int and 0 < count <= 10000, 'Invalid expected source count')
    _require(report.get('status') == 'complete' and report.get('complete') is True
             and report.get('candidate_count') == report.get('processed_count') == count, 'Expected complete processed pair review')
    _require(report.get('training_eligible') is False and report.get('split_modified') is False,
             'Pair review must not approve training or change splits')
    _require(report['intake_sha256'] == dependencies['intake']['sha256']
             and report['preparation_batch_sha256'] == dependencies['prepared_batch']['sha256'], 'Review dependencies differ')
    old = data['previous_pair_report']
    _require(data['previous_pair_summary']['run_report_sha256'] == dependencies['previous_pair_report']['sha256'],
             'Previous summary belongs to another run')
    for key in ('comparison_method', 'thresholds', 'max_side', 'implementation'):
        _require(report[key] == old[key], 'Pair comparison changed from previous reference: '+key)
        if key in config:
            _require(report[key] == config[key], 'Pair comparison differs from registered configuration')
    _require(report['max_side'] == config['max_side'], 'Review resolution differs')
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
    _require(len(full) == config['expected_total_count'] and set(full) == set(queue) and set(items) <= set(full), 'Combined registered inventory differs')
    _require(data['registration_report'].get('status') == 'registered', 'Update was not registered')
    _require(reviewed.get('status') == 'passed' and reviewed['run_report_sha256'] == report_sha
             and reviewed.get('training_eligible') is False, 'Artifact summary is not for this complete run')
    recovery, recovery_sha, recovery_path, used_recovery = {}, None, None, set()
    if semantics_recovery_path is not None:
        recovery_path = Path(semantics_recovery_path).resolve()
        _require(recovery_path.is_relative_to(root), 'Recovery index must stay inside workspace')
        recovery_sha = digest(recovery_path)
        recovered = read_json(recovery_path)
        _require(recovered.get('schema_version') == '1.0' and recovered.get('status') == 'recovered_serialization_only'
                 and recovered.get('run_report_sha256') == report_sha and recovered.get('implementation') == report['implementation']
                 and recovered.get('training_eligible') is False and recovered.get('source_modified') is False,
                 'Recovery index must preserve this run and source semantics')
        recovery = _index(recovered['items'])
        needed = {name for name, pair in pairs.items() if 'semantics' not in pair['artifacts']
                  and any(e.get('stage') == 'semantics' and e.get('type') == 'UnicodeEncodeError' for e in pair['errors'])}
        _require(set(recovery) == needed, 'Recovery inventory differs from exact serialization failures')
    old_inputs = {pair['input_path'] for pair in old['pairs']}
    _require(not old_inputs & set(items) and old_inputs | set(items) == set(full), 'Old and new review identities overlap or leave gaps')
    prepared_root = paths['prepared_batch'].parent
    entries, visual, pair_status, units, display = [], Counter(), Counter(), Counter(), Counter()
    artifacts_seen, sources_seen, target_names = set(), set(), set()
    totals = Counter()
    for ordinal, item in enumerate(intake['items'], 1):
        name = item['input_path']
        pair, prep_item, proof, row = pairs[name], prepared[name], validated[name], rows[name]
        local_id = f'pair-{ordinal:04d}'
        pair_file = run/local_id/'pair.json'
        _require(read_json(pair_file) == pair, 'Local pair report differs from batch record')
        errors = pair.get('errors')
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
        semantics = read_json(artifact_paths['semantics']) if native_semantic else None
        recovery_pin = None
        if name in recovery:
            record = recovery[name]
            _require(not native_semantic and record['input_sha256'] == item['input_sha256']
                     and record['target_path'] == target['path'] and record['target_sha256'] == target['sha256']
                     and record['pair_id'] == local_id and record['method'] == 'json_ensure_ascii_true',
                     'Recovered semantics belongs to a different pair or operation')
            recovered_path = contained(recovery_path.parent, record['semantics']['path'])
            _pin(recovered_path, record['semantics']['sha256'])
            semantics = read_json(recovered_path)
            from .dxf_semantics import audit_dxf_semantics
            _require(semantics == audit_dxf_semantics(contained(dataset, target['path'])),
                     'Recovered semantics differs from a fresh audit of the unchanged original')
            recovery_pin = {'path': recovered_path.relative_to(root).as_posix(), 'sha256': digest(recovered_path)}
            used_recovery.add(name)
            totals['recovered_semantic_reports'] += 1
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
        comparison_available = reference is not None and 'target' in artifact_paths and 'evidence' in pair
        if comparison_available:
            evidence = pair['evidence']
            _require(evidence['method'] == report['comparison_method'] and evidence['status'] == pair['visual_status'], 'Visual status differs from evidence')
            _require(pair['visual_status'] in ('verified', 'quarantine')
                     and pair['status'] == ('review_required' if pair['visual_status'] == 'verified' else 'quarantine'), 'Pair status does not match visual result')
            if 'metrics' in evidence:
                reasons = ['below_'+key for key, threshold in report['thresholds'].items() if evidence['metrics'][key] < threshold]
                _require(evidence['reasons'] == reasons and (evidence['status'] == 'verified') == (not reasons), 'Visual thresholds or reasons disagree')
            else:
                _require(evidence['status'] == 'quarantine' and bool(evidence['reasons']), 'Unregistered visual match cannot be approved')
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
        registered = queue[name]
        _require(registered['input_sha256'] == item['input_sha256'] and registered.get('training_eligible') is False
                 and registered.get('split') is None
                 and registered['candidate_target']['path'] == target['path']
                 and registered['candidate_target']['sha256'] == target['sha256'], 'Registration queue changed source or approval')
        entries.append({'qualified_pair_id': 'update-v1:'+local_id, 'pair_id': local_id,
            'input_path': name, 'input_sha256': item['input_sha256'], 'target_path': target['path'], 'target_sha256': target['sha256'],
            'pair_report': {'path': pair_file.relative_to(root).as_posix(), 'sha256': digest(pair_file)},
            'preparation': {'path': prep_path.relative_to(root).as_posix(), 'sha256': digest(prep_path)},
            'visual_status': pair['visual_status'], 'pair_status': pair['status'],
            'alignment_status': alignment['status'] if alignment else None,
            'dimension_count': len(semantics['dimensions']) if semantics is not None else None,
            'numeric_mismatch_count': len(mismatch) if semantics is not None else None,
            'units': semantics['units'] if semantics is not None else None,
            'units_name': semantics['units_name'] if semantics is not None else None,
            'visual_reasons': pair.get('evidence', {}).get('reasons', []), 'execution_errors': errors,
            'evidence_completeness': 'execution_error' if errors else 'complete',
            'comparison_available': comparison_available,
            'handoff_status': 'technical_quarantine' if errors else pair['status'],
            'raw_semantic_available': native_semantic, 'recovered_semantic_available': recovery_pin is not None,
            'semantic_recovery': recovery_pin,
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
    _require(used_recovery == set(recovery), 'Unused semantics recovery entries')
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
    _pin(config_path, config_sha)
    _pin(report_path, report_sha)
    _pin(summary_path, summary_sha)
    if recovery_path is not None:
        _pin(recovery_path, recovery_sha)
    return {'schema_version': '1.0', 'status': 'verified', 'config_sha256': config_sha,
        'run_report_sha256': report_sha, 'summary_report_sha256': summary_sha,
        'code_sha256': digest(Path(__file__)), 'dependencies': dependencies,
        'semantics_recovery': {'path': recovery_path.relative_to(root).as_posix(), 'sha256': recovery_sha} if recovery_path else None,
        'summary': {'pair_count': len(entries), 'artifact_count': totals['checked_artifacts'],
            'source_files_rehashed': len(sources_seen), 'semantic_dimensions': totals['reviewed_dimension_instances'],
            'numeric_mismatches': totals['reviewed_numeric_mismatches'], 'registrations': totals['invertible_registrations'],
            'container_repaired_sources': totals['container_repaired_sources'], 'visual_status': dict(visual),
            'pairs_with_execution_errors': totals['pairs_with_execution_errors'], 'reference_reports': totals['reference_reports'],
            'raw_semantic_reports': totals['semantic_reports'], 'recovered_semantic_reports': totals['recovered_semantic_reports'],
            'units': dict(units), 'local_html_link_occurrences_verified': len(parser.links)},
        'entries': entries, 'training_eligible': False, 'split_modified': False, 'source_modified': False,
        'limitations': ['No training labels, geometric roles, physical scale or family independence are approved.',
            'The comparison method and thresholds match the previous frozen run.',
            'Composed matrices and anchor evidence are internally consistent; this is not a measurement of registration accuracy.',
            'Source normalization pixels are reproduced from the pinned prepared overview; the original huge RGB is not decoded again.',
            'Cross-pair comparisons are diagnostic probes, not labeled independent negatives.']}
