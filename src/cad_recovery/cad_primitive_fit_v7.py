"""Cheap all-pair screening before bounded exact weak-curvature verification."""
import math

import numpy as np

from .cad_primitive_fit_v6 import Previous, paths, compatible
from .cad_primitive_fit_v2 import POLICY as LIMITS, checked, sector
from .anchored_curve_fit_v1 import circle_solution
from .geometry_detection import _fit_circle
from .stroke_curve_control_v1 import distance_to_polyline

POLICY = dict(version='weak-curvature-family/2.0', minimum_independent_paths=3,
    maximum_hypotheses=LIMITS['maximum_families'], pair_screen_points_per_path=32,
    maximum_screen_pairs=32640, exact_pair_budget=LIMITS['maximum_family_pairs'],
    straight_source_exclusion_px=math.sqrt(.5), missing_sectors_filled=False,
    thresholds='unchanged source p95/max/sensitivity; screen is not final approval')


class Fitter(Previous):
    def families(self, items):
        base = super().families(items)
        audit = dict(policy=POLICY, hypotheses=0, screened_pairs=0, exact_groups=0,
            promoted_items=[], groups=[], search_complete=not self.incomplete)
        base['weak_curvature_discovery'] = audit
        if self.incomplete:
            return base
        hypotheses = {}; weak = set(); samples = {}
        for i, item in enumerate(items):
            p = item['points']; model = item.get('model')
            if model and model['kind'] == 'arc':
                hypotheses[i] = model
            elif (not model or model['kind'] == 'line') and len(p) >= 3 and not np.array_equal(p[0], p[-1]):
                if (np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < LIMITS['minimum_length_px']
                        or distance_to_polyline(p, p[[0, -1]]).max() <= POLICY['straight_source_exclusion_px']):
                    continue
                solution = _fit_circle(p)
                if solution is None:
                    continue
                center, radius, _ = solution
                if not 2 <= radius <= max(self.support.shape)*4:
                    continue
                if sector(p, center, radius) is None:
                    continue
                hypotheses[i] = dict(center_px=center.tolist(), radius_px=float(radius)); weak.add(i)
            if i in hypotheses:
                samples[i] = p[np.linspace(0, len(p)-1, min(len(p), POLICY['pair_screen_points_per_path'])).astype(int)]
        audit['hypotheses'] = len(hypotheses)
        if len(hypotheses) > POLICY['maximum_hypotheses']:
            self.incomplete = True
        ids = sorted(hypotheses); proposals = {}; seen = set()
        screen = np.stack([items[k]['points'][np.linspace(0, len(items[k]['points'])-1,
            POLICY['pair_screen_points_per_path']).astype(int)] for k in ids]) if ids else np.zeros((0, 32, 2))
        for pos, i in enumerate(ids):
            if self.incomplete:
                break
            for j in ids[pos+1:]:
                if i not in weak and j not in weak or not compatible(items, [i, j]):
                    continue
                audit['screened_pairs'] += 1
                if audit['screened_pairs'] > POLICY['maximum_screen_pairs']:
                    self.incomplete = True; break
                solution = _fit_circle(np.r_[samples[i], samples[j]])
                if solution is None:
                    continue
                center, radius, _ = solution
                if not 2 <= radius <= max(self.support.shape)*4:
                    continue
                members = []
                radial = abs(np.linalg.norm(screen-center, axis=2)-radius)
                nearby = [ids[k] for k in np.flatnonzero(radial.max(axis=1) <= LIMITS['max_px'])]
                for k in nearby:
                    if not compatible(items, [i, j, k]):
                        continue
                    old = items[k].get('model')
                    if old and old['kind'] == 'arc' and old.get('locally_stable', True):
                        if (np.linalg.norm(np.asarray(old['center_px'])-center) > LIMITS['max_px']
                                or abs(old['radius_px']-radius) > LIMITS['max_px']):
                            continue
                    # Cheap radial screen, then the exact own walk sector gate.
                    if sector(items[k]['points'], center, radius) is not None:
                        members.append(k)
                key = tuple(members)
                if (i not in members or j not in members or key in seen
                        or len(paths(items, members)) < POLICY['minimum_independent_paths']):
                    continue
                seen.add(key); self.family_pairs += 1; audit['exact_groups'] += 1
                if self.family_pairs > LIMITS['maximum_family_pairs'] or self.calls >= LIMITS['maximum_fit_calls']:
                    self.incomplete = True; break
                self.calls += 1
                refined = circle_solution(np.concatenate([items[k]['points'] for k in members]), True, max(self.support.shape)*4)
                if refined is None or refined[2] > LIMITS['sensitivity_px']:
                    continue
                center, radius, uncertainty = refined
                models = [checked(sector(items[k]['points'], center, radius), items[k]['points'], self.support) for k in members]
                if not all(models):
                    continue
                proposals[key] = dict(members=members, center_px=center.tolist(), radius_px=float(radius),
                    parameter_sensitivity_native_px=float(uncertainty), models=models,
                    independent_source_paths=sorted(paths(items, members)))
        audit['search_complete'] = not self.incomplete
        if self.incomplete:
            audit['reason'] = 'bounded_weak_curvature_search_incomplete_no_new_models_committed'
            return base
        for i in ids:
            options = [group for group in proposals.values() if i in group['members']]
            if not options:
                continue
            best = max(options, key=lambda group: len(group['members']))
            if any(np.linalg.norm(np.asarray(group['center_px'])-best['center_px']) > LIMITS['max_px']
                    or abs(group['radius_px']-best['radius_px']) > LIMITS['max_px'] for group in options):
                continue
            local = best['members'].index(i)
            items[i]['model'] = dict(**best['models'][local], locally_stable=True,
                circle_family_support=best['members'], parameter_sensitivity_native_px=best['parameter_sensitivity_native_px'],
                parameter_evidence='independent_weak_curvature_source_family')
            audit['promoted_items'].append(i)
            if best['members'] not in [group['members'] for group in audit['groups']]:
                audit['groups'].append({key: value for key, value in best.items() if key != 'models'})
        base['groups'].extend(group for group in audit['groups'] if group['members'] not in [old['members'] for old in base['groups']])
        base['pairs'] = self.family_pairs
        return base
