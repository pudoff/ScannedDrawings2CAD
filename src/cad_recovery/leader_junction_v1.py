"""Local native-ink junction proofs without dilation, closing or nominal input."""
import math

import cv2
import numpy as np

from . import angular_association_v1 as core
from . import angular_leader_search_v1 as base
from . import dimension_association_v3 as linear
from .family_catalog_v2 import require

POLICY = {'id': 'leader-junction/1.0', 'max_endpoint_shift_px': 12,
          'roi_half_size_px': 17, 'max_roi_pixels': 1600, 'max_junctions_per_notation': 32,
          'max_junctions_per_page': 128, 'seed_distance_px': 12, 'seed_radius_px': 3,
          'corridor_half_width_px': 3, 'branch_ring_inner_px': 10, 'branch_ring_outer_px': 14,
          'branch_direction_tolerance_deg': 25, 'extra_branch_pixels': 3,
          'pixel_cell_stages': 8, 'connectivity': 8, 'morphological_gap_closing': False,
          'physical_geometry_approved': False, 'production_admitted': False}


class Budget(base.Budget):
    def __init__(self):
        super().__init__()
        self.junction_cache = {}; self.junction_notation = 0
        self.junction_raster = None
        self.junction_roi_pixels = self.junction_pixel_cells = self.junction_cache_hits = 0

    def stats(self):
        return {'distinct_local_junctions': len(self.junction_cache), 'cache_hits': self.junction_cache_hits,
                'native_roi_pixels': self.junction_roi_pixels, 'charged_pixel_cells': self.junction_pixel_cells,
                'accounting': 'Eight fixed mask/coordinate/label/ring passes per native ROI pixel; all ordinary probes charged separately.'}


def prove(shelf, leader, shelf_end, leader_end, raster, budget):
    """Return observed component alternatives; incomplete/extra branches cannot be unique."""
    budget.check()
    require(budget.junction_raster is None or budget.junction_raster is raster, 'Junction cache cannot cross raster/mask frames')
    budget.junction_raster = raster
    key = (shelf['id'], leader['id'], shelf_end, leader_end)
    if key in budget.junction_cache:
        budget.junction_cache_hits += 1
        return budget.junction_cache[key]
    budget.junction_notation += 1
    if (budget.junction_notation > POLICY['max_junctions_per_notation'] or
            len(budget.junction_cache) >= POLICY['max_junctions_per_page']):
        raise core.SearchLimit('local_junction_count_limit')
    point = linear._intersection(shelf, leader)
    near = [shelf['points'][shelf_end], leader['points'][leader_end]]
    if point is None or any(np.linalg.norm(p-point) > POLICY['max_endpoint_shift_px'] for p in near):
        budget.junction_cache[key] = []
        return []
    directions = []
    for line, end in ((shelf, shelf_end), (leader, leader_end)):
        delta = line['points'][1-end]-point; length = float(np.linalg.norm(delta))
        if length < POLICY['seed_distance_px']+POLICY['seed_radius_px']:
            budget.junction_cache[key] = []
            return []
        directions.append(delta/length)
    half = POLICY['roi_half_size_px']; origin = np.floor(point).astype(int)-half
    extent = 2*half+2; area = extent*extent
    if area > POLICY['max_roi_pixels']:
        raise core.SearchLimit('local_junction_roi_limit')
    h, w = raster.gray.shape
    if (origin < 0).any() or (origin+[extent, extent] > [w, h]).any():
        result = [{'point_px': point.tolist(), 'unique_ink_support': False, 'reasons': ['junction_ROI_out_of_frame'],
                   'component_id': None}]
        budget.junction_cache[key] = result
        return result
    charge = area*POLICY['pixel_cell_stages']; budget.cells += charge
    budget.junction_roi_pixels += area; budget.junction_pixel_cells += charge
    if budget.cells > core.POLICY['max_pixel_cells']:
        raise core.SearchLimit('pixel_probe_limit')
    x0, y0 = origin; gray = raster.gray[y0:y0+extent, x0:x0+extent]
    excluded = raster.excluded[y0:y0+extent, x0:x0+extent]
    raw = ((gray < 200) & (excluded == 0)).astype(np.uint8)
    yy, xx = np.mgrid[y0:y0+extent, x0:x0+extent]
    offsets = np.stack((xx-point[0], yy-point[1]), axis=-1)
    distance = np.linalg.norm(offsets, axis=-1)
    node = distance <= POLICY['seed_radius_px']
    corridors = np.zeros(raw.shape, bool); seeds = []
    for unit in directions:
        along = offsets@unit
        across = np.abs(offsets@np.array([-unit[1], unit[0]]))
        corridors |= (across <= POLICY['corridor_half_width_px']) & (along >= -POLICY['seed_radius_px'])
        seed = point+POLICY['seed_distance_px']*unit
        seeds.append((xx-seed[0])**2+(yy-seed[1])**2 <= POLICY['seed_radius_px']**2)
    # Native connectivity is evaluated before any rounding/probe neighbourhood can bridge white pixels.
    _, labels = cv2.connectedComponents((raw & corridors).astype(np.uint8), connectivity=8)
    candidates = set(np.unique(labels[node]))-{0}
    for seed in seeds: candidates &= set(np.unique(labels[seed]))-{0}
    _, full_labels = cv2.connectedComponents(raw, connectivity=8)
    result = []
    for identifier in sorted(candidates):
        connected = labels == identifier
        full_ids = set(np.unique(full_labels[connected]))-{0}
        full = np.isin(full_labels, list(full_ids))
        ring = full & (distance >= POLICY['branch_ring_inner_px']) & (distance <= POLICY['branch_ring_outer_px'])
        explained = np.zeros(raw.shape, bool)
        for unit in directions:
            explained |= offsets@unit >= distance*math.cos(math.radians(POLICY['branch_direction_tolerance_deg']))
        extras = int(np.count_nonzero(ring & ~explained))
        reasons = []
        if extras >= POLICY['extra_branch_pixels']: reasons.append('unassigned_ink_branch')
        if len(candidates) != 1: reasons.append('competing_native_components')
        node_pixels = int(np.count_nonzero(connected & node))
        result.append({'point_px': point.tolist(), 'component_id': int(identifier),
                       'unique_ink_support': not reasons, 'reasons': reasons,
                       'endpoint_shifts_px': [float(np.linalg.norm(p-point)) for p in near],
                       'roi_box_px': [int(x0), int(y0), int(x0+extent), int(y0+extent)],
                       'native_component_pixels': int(np.count_nonzero(connected)), 'node_pixels': node_pixels,
                       'branch_seed_pixels': [int(np.count_nonzero(connected & s)) for s in seeds],
                       'unassigned_ring_pixels': extras, 'morphological_gap_closing': False})
    budget.junction_cache[key] = result
    return result
