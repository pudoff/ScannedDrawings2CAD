"""Corroborate weak source curvature across independent observed walks.

No isolated weak LINE is converted to ARC. New circle hypotheses need at
least three source paths, stable joint parameters, own-sector checks and a
unique explanation within the existing fit error. Missing angles stay missing.
"""
from copy import deepcopy
import math

import numpy as np

from .cad_primitive_fit_v5 import Fitter as Previous
from .cad_primitive_fit_v2 import POLICY as LIMITS, checked, sector
from .anchored_curve_fit_v1 import circle_solution
from .stroke_curve_control_v1 import distance_to_polyline

POLICY = dict(version='weak-curvature-family/1.0', minimum_independent_paths=3,
    maximum_hypotheses=LIMITS['maximum_families'],
    straight_source_exclusion_px=math.sqrt(.5),
    fitting_thresholds='unchanged cad-primitive-fit/2.0', missing_sectors_filled=False)


def paths(items, members):
    return {path for i in members for path in items[i]['path_ids']}


def compatible(items, members):
    return len(set().union(*(items[i]['roles'] for i in members))) <= 1


class Fitter(Previous):
    def families(self, items):
        base = super().families(items)
        initial = deepcopy(items)
        audit = dict(policy=POLICY, hypotheses=0, promoted_items=[], groups=[],
            independent_path_requirement=True, search_complete=not self.incomplete)
        base['weak_curvature_discovery'] = audit
        if self.incomplete:
            return base
        hypotheses = {}; weak = set()
        for i, item in enumerate(items):
            model = item.get('model'); p = item['points']
            if model and model['kind'] == 'arc':
                hypotheses[i] = model
                continue
            if model and model['kind'] not in ('line',):
                continue
            if (len(p) < 3 or np.array_equal(p[0], p[-1])
                    or np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < LIMITS['minimum_length_px']
                    or distance_to_polyline(p, p[[0, -1]]).max() <= POLICY['straight_source_exclusion_px']):
                continue
            if self.calls >= LIMITS['maximum_fit_calls']:
                self.incomplete = True; break
            self.calls += 1
            solution = circle_solution(p, False, max(self.support.shape)*4)
            if solution is None:
                continue
            center, radius, uncertainty = solution
            candidate = sector(p, center, radius)
            if candidate is None:
                continue
            hypotheses[i] = dict(**candidate, parameter_sensitivity_native_px=float(uncertainty))
            weak.add(i)
        audit['hypotheses'] = len(hypotheses)
        if len(hypotheses) > POLICY['maximum_hypotheses']:
            self.incomplete = True
        proposals = {}; ids = sorted(hypotheses)
        for pos, i in enumerate(ids):
            if self.incomplete:
                break
            a = hypotheses[i]
            for j in ids[pos+1:]:
                if i not in weak and j not in weak or not compatible(items, [i, j]):
                    continue
                b = hypotheses[j]
                uncertainty = a.get('parameter_sensitivity_native_px', 2)+b.get('parameter_sensitivity_native_px', 2)+2
                if (np.linalg.norm(np.asarray(a['center_px'])-b['center_px']) > uncertainty
                        or abs(a['radius_px']-b['radius_px']) > uncertainty):
                    continue
                self.family_pairs += 1
                if self.family_pairs > LIMITS['maximum_family_pairs'] or self.calls >= LIMITS['maximum_fit_calls']:
                    self.incomplete = True; break
                self.calls += 1
                solution = circle_solution(np.r_[items[i]['points'], items[j]['points']], True, max(self.support.shape)*4)
                if solution is None:
                    continue
                center, radius, _ = solution; members = []
                for k in ids:
                    if not compatible(items, [i, j, k]):
                        continue
                    old = items[k].get('model')
                    if old and old['kind'] == 'arc' and old.get('locally_stable', True):
                        if (np.linalg.norm(np.asarray(old['center_px'])-center) > LIMITS['max_px']
                                or abs(old['radius_px']-radius) > LIMITS['max_px']):
                            continue
                    if sector(items[k]['points'], center, radius) is not None:
                        members.append(k)
                key = tuple(members)
                if (i not in members or j not in members or key in proposals
                        or len(paths(items, members)) < POLICY['minimum_independent_paths']):
                    continue
                if self.calls >= LIMITS['maximum_fit_calls']:
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
        groups = []
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
                circle_family_support=best['members'],
                parameter_sensitivity_native_px=best['parameter_sensitivity_native_px'],
                parameter_evidence='independent_weak_curvature_source_family')
            audit['promoted_items'].append(i)
            if best['members'] not in [group['members'] for group in groups]:
                groups.append({key: value for key, value in best.items() if key != 'models'})
        # All retained evidence is in the original item domain used by the
        # assembler's source-edge ledger. No new raster walk or edge is created.
        assert all(np.array_equal(before['points'], after['points']) for before, after in zip(initial, items))
        audit['groups'] = groups
        base['groups'].extend(group for group in groups if group['members'] not in [old['members'] for old in base['groups']])
        base['pairs'] = self.family_pairs
        return base
