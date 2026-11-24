"""Complete a pinned partial raster batch using explicit container repair cases.

The partial run is preserved. Successful directories are copied byte for byte;
failed sources can only use a pinned repair case in a new output directory.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import time

from .clear_pairs import contained, digest, read_json
from .object_review_case import _pinned, _write_json
from .png_container_repair import prepare_repaired_opaque_rgba
from .updated_raster_batch import _checkpoint, _finish_index, _header, _peak_memory


def _tree_pins(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Preparation copy may not contain symbolic links')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = digest(path)
    if not result:
        raise ValueError('Cannot reuse an empty preparation directory')
    return result


def _verify_preparation(path, source_sha, depth=0):
    if depth > 2:
        raise ValueError('Preparation provenance nesting exceeds reuse limit')
    report = read_json(path)
    if (report.get('status') != 'prepared' or report.get('source_integrity_validated') is not True
            or report.get('source_modified') is not False or report.get('source_sha256') != source_sha):
        raise ValueError('Cannot reuse incomplete preparation or changed source identity')
    for item in [report['overview'], report['ink_overview'], *report['tiles']]:
        artifact = contained(path.parent, item['path'])
        _pinned(artifact, item['sha256'])
        if artifact.stat().st_size != item['size_bytes']:
            raise ValueError('Prepared image byte count differs from its pinned report')
    if 'rgba_conversion' in report:
        conversion = report['rgba_conversion']
        derivative = conversion['rgb_derivative']
        _pinned(contained(path.parent, derivative['path']), derivative['sha256'])
        nested = conversion['rgb_preparation_report']
        nested_path = contained(path.parent, nested['path'])
        _pinned(nested_path, nested['sha256'])
        _verify_preparation(nested_path, derivative['sha256'], depth+1)


def complete_prepared_update(workspace: Path, config_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    root, config_path, output = Path(workspace).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if (not config_path.is_relative_to(root) or not output.is_relative_to(root)
            or output == root or output.exists()):
        raise ValueError('Choose a new output within workspace')
    config_sha = digest(config_path)
    config = read_json(config_path)
    if config.get('schema_version') != '1.0' or config.get('purpose') != 'explicit_container_repair_completion':
        raise ValueError('Expected explicit container repair configuration')
    dataset = contained(root, config['dataset'])
    predecessor_path = contained(root, config['predecessor_report']['path'])
    predecessor_dir = predecessor_path.parent
    intake_path = contained(root, config['intake']['path'])
    if (output.is_relative_to(dataset) or output.is_relative_to(predecessor_dir)
            or predecessor_dir.is_relative_to(output)):
        raise ValueError('Output must be separate from source and previous preparation')
    _pinned(predecessor_path, config['predecessor_report']['sha256'])
    _pinned(intake_path, config['intake']['sha256'])
    previous, intake = read_json(predecessor_path), read_json(intake_path)
    if (previous.get('status') != 'partial' or previous.get('purpose') != 'raster_preparation_only'
            or previous.get('training_eligible') is not False or previous.get('source_modified') is not False
            or previous.get('intake_sha256') != config['intake']['sha256']
            or intake.get('training_eligible') is not False):
        raise ValueError('Expected a pinned, unapproved partial preparation')
    manifest = read_json(predecessor_dir/'manifest.json')
    for name, expected in manifest['artifact_sha256'].items():
        _pinned(contained(predecessor_dir, name), expected)
    code = dict(previous['implementation'])
    for name, expected in code.items():
        _pinned(contained(Path(__file__).parent, name), expected)
    code.update({name: digest(Path(__file__).with_name(name))
                 for name in ['repaired_raster_batch.py', 'png_container_repair.py']})
    inputs = {item['input_path']: item for item in intake['items']}
    records = {item['input_path']: item for item in previous['items']}
    if (len(inputs) != len(intake['items']) or len(records) != len(previous['items'])
            or set(inputs) != set(records) or len(inputs) != config['expected_count']
            or any(item['status'] not in ('prepared', 'failed') for item in records.values())):
        raise ValueError('Partial batch inventory differs from registered intake')
    repairs = {case['source_sha256']: case for case in config['repair_cases']}
    failed = [item for item in records.values() if item['status'] == 'failed']
    if (not repairs or len(repairs) != len(config['repair_cases']) or len(failed) != len(repairs)
            or {item['input_sha256'] for item in failed} != set(repairs)):
        raise ValueError('Every failed input needs exactly one explicit repair case')
    source_paths, output_paths, reuse_pins = set(), set(), {}
    for name, item in inputs.items():
        source, record = contained(dataset, name), records[name]
        destination = contained(output, record['output_directory'])
        if source in source_paths or destination in output_paths or destination.parent != output:
            raise ValueError('Duplicate source or invalid preparation directory')
        source_paths.add(source)
        output_paths.add(destination)
        if (item.get('training_eligible') is not False or item.get('split') is not None
                or record['input_sha256'] != item['input_sha256']
                or source.stat().st_size != item['input_size_bytes']):
            raise ValueError('Source metadata or training state differs from intake')
        _pinned(source, item['input_sha256'])
        width, height, color = _header(source)
        if color != record['input_color_type']:
            raise ValueError('Source encoding differs from previous batch')
        if item.get('candidate_target'):
            target = item['candidate_target']
            _pinned(contained(dataset, target['path']), target['sha256'])
        if record['status'] == 'prepared':
            directory = contained(predecessor_dir, record['output_directory'])
            prep_path = contained(predecessor_dir, record['preparation_report'])
            if directory.is_symlink() or prep_path != directory/'preparation.json':
                raise ValueError('Prepared report must belong to its source directory')
            _pinned(prep_path, record['preparation_report_sha256'])
            _verify_preparation(prep_path, item['input_sha256'])
            metadata = read_json(prep_path)
            if (metadata['source_size_px'] != [width, height]
                    or record.get('source_size_px') != [width, height]
                    or metadata['tile_count'] != len(metadata['tiles'])
                    or record.get('tile_count') != len(metadata['tiles'])):
                raise ValueError('Prepared dimensions or tile count differ from source inventory')
            reuse_pins[name] = _tree_pins(directory)
        elif record['input_color_type'] != 6:
            raise ValueError('Explicit repair currently supports opaque RGBA only')
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'config.json', config)
    report = deepcopy(previous)
    report.update(status='running', config_sha256=config_sha, implementation=code,
                  items=[], container_repair_cases=config['repair_cases'],
                  lineage={'predecessor_report': config['predecessor_report'],
                           'predecessor_status': previous['status'],
                           'predecessor_config_sha256': previous['config_sha256'],
                           'reused_sources': 0, 'repaired_sources': 0,
                           'copied_files': 0, 'copy_sha256_verified': True},
                  limitations=[*previous['limitations'],
                      'One explicitly pinned PNG prefix may represent a nonconformant original container.',
                      'Excluded trailer bytes are preserved separately; original dataset bytes are unchanged.',
                      'A complete preparation report still requires independent output validation.'])
    report['counts'] = {**previous['counts'], 'prepared': 0, 'failed': 0,
                        'rgb_prepared': 0, 'opaque_rgba_prepared': 0}
    report.pop('resources', None)
    _checkpoint(output, report)
    try:
        for name, original in records.items():
            record = deepcopy(original)
            destination = contained(output, record['output_directory'])
            if original['status'] == 'prepared':
                shutil.copytree(contained(predecessor_dir, original['output_directory']), destination)
                if _tree_pins(destination) != reuse_pins[name]:
                    raise ValueError('Copied preparation differs from original successful output')
                report['lineage']['reused_sources'] += 1
                report['lineage']['copied_files'] += len(reuse_pins[name])
            else:
                params = {key: previous['parameters'][key]
                          for key in ['tile_size', 'overlap', 'overview_max_side', 'strip_rows']}
                result = prepare_repaired_opaque_rgba(contained(dataset, name), destination,
                    repair_case=repairs[original['input_sha256']], **params)
                if result['status'] != 'prepared' or result['source_sha256'] != original['input_sha256']:
                    raise ValueError('Repair did not produce a complete original-source wrapper')
                record.pop('error', None)
                record.update(status='prepared', preparation_report=record['output_directory']+'/preparation.json',
                              preparation_report_sha256=digest(destination/'preparation.json'),
                              source_size_px=result['source_size_px'], tile_count=len(result['tiles']),
                              container_repaired=True)
                report['lineage']['repaired_sources'] += 1
            report['items'].append(record)
            report['counts']['prepared'] += 1
            report['counts']['rgb_prepared' if record['input_color_type'] == 2 else 'opaque_rgba_prepared'] += 1
        for name, expected in code.items():
            _pinned(Path(__file__).with_name(name), expected)
        _pinned(config_path, config_sha)
        _pinned(predecessor_path, config['predecessor_report']['sha256'])
        _pinned(intake_path, config['intake']['sha256'])
        for name, item in inputs.items():
            _pinned(contained(dataset, name), item['input_sha256'])
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', completion_error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started, 4)
        report['resources'] = _peak_memory()
        _checkpoint(output, report)
        _finish_index(output, report)
    return report
