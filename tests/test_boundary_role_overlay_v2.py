from copy import deepcopy
import numpy as np
import pytest

from cad_recovery.boundary_role_overlay_v1 import analyze as old_analyze
from cad_recovery import boundary_role_overlay_v2 as guarded
from test_boundary_role_overlay_v1 import packet as sparse_packet


def packet():
    args = sparse_packet()
    for interval in args[1]['proposals']:
        vertices = np.asarray(interval['points_px'], float)
        p = [vertices[0]]
        for a, b in zip(vertices, vertices[1:]):
            n = int(np.ceil(np.linalg.norm(b-a)))
            p.extend(a+(b-a)*t for t in np.linspace(0, 1, n+1)[1:])
        interval['points_px'] = np.asarray(p).tolist()
        interval['source_index_interval'] = [0, len(p)-1]
    return args


def test_guard_abstains_at_observed_contacts_and_keeps_remote_part():
    args = packet(); before = old_analyze(*args); after = guarded.analyze(*args)
    assert 0 < after['added_pixels'] < before['added_pixels']
    assert after['competitor_guard']['removed_pixels'] > 0 and not after['search_incomplete']
    assert after['pixel_roles'][80, 30] == 255 and after['pixel_roles'][150, 100] == 1
    assert np.all((after['pixel_roles'] == args[2]) | (after['pixel_roles'] == before['pixel_roles']))


@pytest.mark.parametrize('field', ['source_points', 'pair_checks'])
def test_audit_is_bounded_and_serializable(field):
    result = guarded.analyze(*packet())
    assert 0 < result['competitor_guard'][field] <= 2000000
    assert sum(r['added_pixels'] for r in result['interval_admission']) == result['added_pixels']


def test_guard_preserves_inherited_roles_and_all_inputs():
    gray, previous, base, local = packet(); base[80, 60:90] = 3
    snapshot = deepcopy(previous); old_gray = gray.copy(); old_base = base.copy()
    result = guarded.analyze(gray, previous, base, local)
    assert np.array_equal(result['pixel_roles'][base != 255], base[base != 255])
    assert previous == snapshot and np.array_equal(gray, old_gray) and np.array_equal(base, old_base)


def test_contact_extent_follows_measured_source_width():
    gray, previous, base, local = packet()
    a = guarded.analyze(gray, previous, base, local)
    radii = [r['competitor_guard']['contact_radius_px'] for r in a['interval_admission'] if r.get('competitor_guard')]
    assert radii and all(v > 2**.5/2 for v in radii)
    assert not a['semantic_ownership_confirmed'] and not a['physical_geometry_approved'] and not a['production_admitted']


def test_no_chain_cannot_create_part_roles():
    gray, previous, base, local = packet(); local['hybrid']['dimension_chains'] = []
    result = guarded.analyze(gray, previous, base, local)
    assert result['added_pixels'] == 0


def test_too_few_competitor_width_samples_abstains_explicitly():
    result = guarded.analyze(*sparse_packet())
    assert result['search_incomplete'] and result['added_pixels'] == 0
