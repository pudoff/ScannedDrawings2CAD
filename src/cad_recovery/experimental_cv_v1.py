"""Fresh, bounded demo inference using the frozen experimental algorithms.

No dataset, reviewed labels, cached experiment predictions or acceptance decisions
enter this adapter. Automatic proposals remain unverified.
"""
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from .dataset_layout_v1 import pin
from .pipeline import write_json

ROOT = Path(__file__).resolve().parents[2]
MAX_PIXELS = 4_000_000
MAX_SIDE = 2000


def analyze(gray_path, folder):
    # Reuse the exact pixel-only experiment producers, not their evaluation jobs.
    for dependency in (ROOT/'scripts', ROOT/'artifacts/cv-skeleton-runtime-v1/deps'):
        if str(dependency) not in sys.path:
            sys.path.insert(0, str(dependency))
    from source_chain_inputs_v1 import prepare
    from run_head_recall_v1 import infer
    from .source_chain_profile_v2 import analyze as routes, diagnostic_heads
    from .head_junction_ownership_v1 import apply
    from .open_reference_chains_v2 import extend
    from .open_reference_context_v2 import filter_context
    from .joint_cv_profile_v1 import analyze as joint
    from .grid_role_revision_v1 import cv_authority, sha
    from .source_region_roles_v1 import analyze as regions
    from .part_admission_v1 import analyze as part, provenance
    from .page_frame_context_v2 import detect
    from .frame_grid_guard_v2 import bind_context
    from .frame_grid_guard_v1 import analyze as coarse_guard
    from .native_frame_v1 import source_masks, refine

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with Image.open(gray_path) as source:
        original_size = list(source.size)
        image = source.convert('L')
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    image.save(folder/'gray.png')
    identity = np.eye(3).tolist()
    view = {'gray': pin(ROOT, (folder/'gray.png').relative_to(ROOT).as_posix()),
            'mode': 'native_crop', 'input_to_native': identity}
    gray, source, geometry, inherited, graph, local, old_heads, extra, calls = prepare(ROOT, view)
    routed = routes(gray, graph, inherited, local['local']['text_mask'], old_heads, diagnostic_heads=extra)
    advanced = diagnostic_heads(gray, old_heads)
    raw_heads = advanced['candidates']
    raw = routes(gray, graph, routed['pixel_roles'], local['local']['text_mask'], raw_heads, diagnostic_heads=extra)
    base = apply(gray, graph, routed['pixel_roles'], raw, raw_heads)
    localized, context, detector, heads = infer(gray, raw_heads, graph['tracks'])
    allowed = set(base['head_ids_used'])
    original_ids = {h['id'] for h in localized['localized']}
    effective = [h for h in heads['guarded'] if h['id'] in allowed or h['id'] not in original_ids]
    links = filter_context(gray, graph, base, extend(gray, graph, base, effective))
    fullbase = {**base, 'dimension_chains': base['dimension_chains']+links['open_reference_chains']}
    packet = joint(gray, graph, fullbase, heads['previous'], heads['guarded'],
                   prior_binding_ids=allowed & {h['id'] for h in heads['previous']})
    roles = packet['joint']['pixel_roles']
    Image.fromarray(roles).save(folder/'raw-roles.png')
    authority = cv_authority(gray, roles, pin(ROOT, (folder/'raw-roles.png').relative_to(ROOT).as_posix()))
    proposed = regions(gray, graph, roles, authority=authority)
    previous, candidate, owners = proposed['previous'], proposed['guarded'], proposed['owners']['guarded']
    Image.fromarray(candidate).save(folder/'candidate-roles.png')
    evidence = provenance(gray, graph, roles, previous, candidate, owners,
        packet['joint']['dimension_chains'], pin(ROOT, (folder/'candidate-roles.png').relative_to(ROOT).as_posix()))
    no_part = part(gray, graph, roles, previous, candidate, owners, packet['joint']['dimension_chains'],
                   authority=authority, evidence=evidence)['no_part']
    height, width = gray.shape
    page = {'kind': 'authenticated_page_overview', 'source_verified': True,
        'source_size_px': [width, height], 'size_px': [width, height], 'patch_bbox': [0, 0, width, height],
        'source_to_patch': identity, 'patch_to_source': identity, 'overview_to_source': identity,
        'source': view['gray']}
    frame = detect(gray)
    binding = bind_context(gray, graph, previous, no_part, owners, page, frame, sha(gray), identity, view['gray'])
    coarse = coarse_guard(gray, graph, roles, previous, no_part, owners, authority=authority, context=binding)
    masks = source_masks(gray, page, frame, context_gray=gray)
    native = refine(gray, graph, roles, previous, no_part, coarse['page_context'], owners,
                    authority=authority, input_to_native=identity, source_gray=gray, source=masks)
    result_roles = native['native_mid']
    complete = not any((graph.get('search_incomplete', False), routed['search_incomplete'],
        advanced['search_incomplete'], localized['search_incomplete'], not context['complete'],
        not detector['complete'], links['search_incomplete'], packet['joint']['search_incomplete'],
        not proposed['complete'], not coarse['complete'], not native['complete']))
    summary = {'version': 'experimental-cv-demo/1.0', 'role_arm': 'native_mid',
        'size_px': [width, height], 'input_size_px': original_size,
        'resized': original_size != [width, height], 'search_complete': complete,
        'geometry_count': len(geometry), 'head_count': len(packet['width_contract']['normalized_heads']),
        'dimension_chain_count': len(packet['joint']['dimension_chains']),
        'role_pixels': {str(int(k)): int(v) for k, v in zip(*np.unique(result_roles, return_counts=True))},
        'ground_truth_used': False, 'cached_predictions_used': False,
        'physical_geometry_approved': False, 'production_admitted': False,
        'coordinate_system': 'experimental_working_raster_pixel_centres',
        'input_to_working': [[width/original_size[0],0,(width/original_size[0]-1)/2],
                            [0,height/original_size[1],(height/original_size[1]-1)/2],[0,0,1]]}
    Image.fromarray(result_roles).save(folder/'roles.png')
    write_json(folder/'graph.json', graph)
    write_json(folder/'geometry.json', geometry)
    write_json(folder/'heads.json', packet['width_contract']['normalized_heads'])
    write_json(folder/'chains.json', packet['joint']['dimension_chains'])
    write_json(folder/'summary.json', summary)
    # Diagnostic head detections alone must not become large filled CAD arrows.
    linked_ids = {h for chain in packet['joint']['dimension_chains'] for h in chain.get('head_ids', [])}
    used_ids = set(packet['joint'].get('head_ids_used', [])) & linked_ids
    return {'gray': gray, 'roles': result_roles, 'geometry': geometry,
            'heads': [h for h in packet['width_contract']['normalized_heads'] if h['id'] in used_ids],
            'summary': summary}
