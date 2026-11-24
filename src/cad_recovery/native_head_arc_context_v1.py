"""A registered native filled-head apex can explain one radial annotation arm."""
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked, read
from .grid_role_revision_v1 import sha


def load_context(record, gray):
    if record is None:
        return None
    root = Path(__file__).resolve().parents[2]
    with Image.open(checked(root, record['working_gray'])) as image:
        observed = np.array(image.convert('L'))
    if sha(observed) != sha(gray):
        raise ValueError('Native head context belongs to another working source')
    audit = read(checked(root, record['qualification']))
    checked(root, audit['source'])
    for ref in audit['code']+audit['provenance']:
        checked(root, ref)
    for row in audit['native_windows']:
        checked(root, row['file'])
    return dict(heads=[row for row in audit['heads'] if row['status'] == 'keep_native'
        and row['native_fill'] >= .7 and min(row['context_support']) >= .8],
        source=audit['source'], qualification=record['qualification'])


def explain(items, i, j, p, q, evidence):
    if not evidence:
        return False, [], None
    low = np.minimum(p, q); high = np.maximum(p, q); fragments = []; arms = []
    for k, item in enumerate(items):
        if k in (i, j):
            continue
        points = item['points']
        near = (np.max(abs(points-p), axis=1) <= 1) | (np.max(abs(points-q), axis=1) <= 1)
        if not near.any():
            continue
        if ((points >= low) & (points <= high)).all():
            fragments.append(k)
        else:
            arms.append(k)
    if len(arms) != 1:
        return False, [], None
    carrier = items[arms[0]]; line = carrier.get('model')
    if not line or line['kind'] != 'line':
        return False, [], None
    ends = np.asarray(line['points_px'], float); direction = ends[1]-ends[0]; length = np.linalg.norm(direction)
    if length < 8:
        return False, [], None
    direction /= length; center = np.asarray(items[i]['model']['center_px'])
    delta = center-ends[0]
    if abs(direction[0]*delta[1]-direction[1]*delta[0]) > 2:
        return False, [], None
    matches = []
    for head in evidence['heads']:
        triangle = np.asarray(head['polygon_px']); cosines = []
        for v in range(3):
            a, b = triangle[(v+1)%3]-triangle[v], triangle[(v+2)%3]-triangle[v]
            cosines.append(float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        apex_index = int(np.argmax(cosines)); apex = triangle[apex_index]
        axis = apex-np.mean(np.delete(triangle, apex_index, axis=0), axis=0)
        if (np.linalg.norm(apex-(p+q)/2) <= 2 and np.linalg.norm(axis) > 0
                and abs(float(axis@direction))/np.linalg.norm(axis) >= .98):
            matches.append(head)
    if len(matches) != 1:
        return False, [], None
    return True, fragments, dict(head_id=matches[0]['id'], source=evidence['source'],
        qualification=evidence['qualification'], carrier_path_ids=carrier['path_ids'],
        role_assignment_claimed=False, physical_geometry_approved=False)
