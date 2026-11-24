"""Native-source adapter to the existing semantic CAD components; CLI candidate.

Cancellation and limits leave a private failure record. No incomplete page is
published under a successful run report. Per-window refusals stay explicit.
"""
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .dataset_layout_v1 import digest, encoded
from .native_sheet_source_v1 import prepare, NativePage, tiles
from .native_cad_compose_v1 import Composer
from .native_cad_window_v1 import run_window
from .raster_input_v2 import AreaOverview, target_size
from .grid_role_revision_v1 import sha

POLICY = dict(core_side=1024, halo=128, max_seconds=3600,
              max_disk_bytes=24*1024**3, max_page_entities=500000,
              max_geometry_observations=200000, overview_side=2000,
              foreground='owned-window Otsu union original gray<240',
              resampling='none on native windows; overview only for global context',
              page_routing='global observation only; geometry always retained',
              crop_frame_grid_authority='withdrawn unless crop is the entire page',
              source_ink='switchable DXF source cores and original grayscale surface',
              production_admitted=False, physical_geometry_approved=False)


def _write(path, value):
    path.write_bytes(encoded(value))


def global_context(page, folder):
    from .page_frame_context_v3 import detect
    from .page_frame_semantics_v1 import POLICY as frame_policy
    from .sheet_router_v1 import observe, route
    accumulator = AreaOverview(page.width, page.height, target_size(page.width, page.height, POLICY['overview_side']))
    for y in range(0, page.height, 64):
        accumulator.add(y, page.crop([0, y, page.width, min(page.height, y+64)]))
    gray = accumulator.finish()
    Image.fromarray(gray).save(folder/'whole-page.png')
    context = dict(size_px=list(gray.shape[::-1]), source_size_px=[page.width, page.height],
        pixel_sha256=sha(gray), whole_page_file_sha256=digest(folder/'whole-page.png'),
        frame_detector=detect(gray), policy=frame_policy, route=route(observe(gray)),
        geometry_enabled=True, table_branch_enabled=False,
        automatic_page_kind_admitted=False, overview_is_not_native_geometry=True)
    _write(folder/'global-context.json', context)
    return context


def run(source, output, *, engine, cancel=None, core=1024, halo=128, _window_runner=None):
    from .profile_contract_v6 import resolve_profile
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    # Bound the real CV core below its 2000px thumbnail limit, including halos.
    if type(core) is not int or type(halo) is not int:
        raise ValueError('Native CAD tile dimensions must be integers')
    if not 64 <= core <= 1536 or not 0 <= halo <= 128:
        raise ValueError('Native CAD requires integer core64..1536 and halo0..128')
    output.mkdir(parents=True)
    started = time.perf_counter()
    completed = []
    state = dict(schema_version='native-cad-pipeline/1.0', state='running',
                 policy=POLICY, core=core, halo=halo, completed_pages=0, completed_windows=0,
                 component_binding=resolve_profile('native_cad_layers_draft'),
                 replaces_current_profile=False, optimizer_steps=0)
    _write(output/'execution.json', state)

    def check():
        if cancel and cancel(): raise InterruptedError('Native CAD cancelled; private artifacts retained')
        if time.perf_counter()-started > POLICY['max_seconds']:
            raise TimeoutError('Native CAD time budget reached')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > POLICY['max_disk_bytes']:
            raise ValueError('Native CAD private disk quota reached')

    try:
        check()
        prepared = prepare(source, output/'source')
        for i, record in enumerate(prepared['pages']):
            check()
            folder = output/f'page-{i+1:03d}'
            folder.mkdir()
            page = NativePage(record)
            try:
                context = global_context(page, folder)
                composer = Composer([page.width, page.height], prepared['source']['sha256'])
                coverage = []
                for tile in tiles(page.width, page.height, core, halo):
                    check()
                    gray = page.crop(tile['box'])
                    window = folder/f"tile-{tile['index']:04d}"
                    if gray.min() == 255:
                        window.mkdir()
                        report = dict(core_completed=True, blank=True, CAD=None,
                                      reason=None, source_sha256=sha(gray))
                    else:
                        report = (_window_runner or run_window)(gray, window, tile, context, engine)
                    composer.add(gray, tile, report)
                    proof = dict(**tile, source_pixel_sha256=sha(gray), size_px=list(gray.shape[::-1]),
                        resize_applied=False, input_to_global=[[1, 0, tile['box'][0]],
                            [0, 1, tile['box'][1]], [0, 0, 1]], **report)
                    _write(window/'coverage.json', proof)
                    coverage.append(proof)
                    state['completed_windows'] += 1
                    _write(output/'execution.json', state)
                    if len(composer.model) > POLICY['max_page_entities']:
                        raise ValueError('Native CAD entity quota reached')
                    if sum(len(v) for v in composer.geometry.values()) > POLICY['max_geometry_observations']:
                        raise ValueError('Native CAD observation quota reached')
                    print('Native CAD page', i+1, 'window', tile['index']+1,
                          'complete', report['core_completed'], flush=True)
                check()
                temporary = folder/'private-page.dxf'
                result = composer.finish(page, temporary)
                expected_area = page.width*page.height
                owned_area = sum((r['core'][2]-r['core'][0])*(r['core'][3]-r['core'][1]) for r in coverage)
                if owned_area != expected_area:
                    raise ValueError('Owned source coverage is incomplete')
                result.update(page_index=i, coverage=coverage, pixel_coverage_complete=True,
                    core_search_complete=all(r['core_completed'] for r in coverage),
                    incomplete_windows=[r['index'] for r in coverage if not r['core_completed']],
                    source_orientation=record['orientation'], global_context=context,
                    raw_source_to_oriented=page.matrix.tolist(),
                    oriented_to_DXF=[[1, 0, 0], [0, -1, page.height-1], [0, 0, 1]],
                    source_surface_sha256=record['surface_sha256'])
                _write(folder/'report.json', result)
                completed.append(result)
                state['completed_pages'] += 1
                _write(output/'execution.json', state)
                # Page artifacts remain private until ALL pages have finished.
            finally:
                page.close()
        check()
        if digest(source) != prepared['source']['sha256']:
            raise ValueError('Native CAD original source changed')
        for i in range(len(completed)):
            (output/f'page-{i+1:03d}'/'private-page.dxf').replace(output/f'page-{i+1:03d}.dxf')
        state.update(state='complete_with_review_required', elapsed_seconds=time.perf_counter()-started,
                     source=prepared['source'], pages=completed,
                     semantic_quality_accepted=False, production_admitted=False,
                     native_windows_complete=True,
                     semantic_core_search_complete=all(p['core_search_complete'] for p in completed),
                     limitations=['Raw oriented windows; no independent tile deskew or global resampling.',
                         'Crop FRAME/GRID authority withdrawn; global frame uses the whole-page overview.',
                         'Incomplete CV windows preserve all foreground ink and raw OCR observations.',
                         'Cross-window unbound dimension chains are not invented; residual ink is retained.',
                         'Polyline crossings remain explicit clipped draft fragments.',
                         'Conflicting OCR readings and unresolved joins require review.',
                         'Full-page native quality and CLI/API/worker admission remain10.4.3.3.'])
        _write(output/'report.json', state)
        _write(output/'execution.json', state)
        return state
    except BaseException as error:
        state.update(state='cancelled' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                     error=str(error), accepted=False, elapsed_seconds=time.perf_counter()-started)
        _write(output/'failure.json', state)
        _write(output/'execution.json', state)
        raise
