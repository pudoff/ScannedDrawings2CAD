"""Check explicitly reviewed diameter feet at material-contour junctions.

An unobserved circle is not manufactured from the nominal OCR value. Two
opposed observed heads define the draft diameter, while independent material
paths corroborate both contact feet. Physical scale remains unconfirmed.
"""
import math
import numpy as np
from .cad_full_diameter_v1 import validate as validate_full_diameters


def validate(packet,annotations):
    validate_full_diameters(packet,annotations)
    lookup={c['id']:c for c in packet['geometry']};heads={h['id']:h for h in annotations['heads']}
    all_shafts={i for row in annotations['dimensions'] for i in row.get('source_candidate_ids',[])}
    for row in annotations['dimensions']:
        contacts=row.get('material_contact_review')
        if contacts is None:continue
        if row.get('evidence_mode') not in ('agent_reviewed_draft','user_reviewed_draft') or not row.get('reviewer'):
            raise ValueError('Material contacts require explicit source review')
        if row['kind']!='diameter' or len(contacts)!=2:raise ValueError('Two diameter material contacts required')
        center,tip=np.asarray(row['points_px'],float);radius=np.linalg.norm(tip-center)
        actual_tips=[np.asarray(heads[i]['tip_px'],float) for i in row['head_ids']];seen=set()
        for contact in contacts:
            foot=contact['foot_candidate_id'];materials=contact['material_candidate_ids']
            if foot not in row['anchor_source_candidate_ids'] or foot not in lookup:
                raise ValueError('Contact foot is not a bound anchor')
            if len(materials)<2 or len(set(materials))!=len(materials) or set(materials)-lookup.keys() or set(materials)&all_shafts:
                raise ValueError('Independent material paths required')
            p=np.asarray(lookup[foot]['support_points'],float)
            if np.max(abs(np.linalg.norm(p-center,axis=1)-radius))>2:raise ValueError('Contact outside defining diameter')
            distances=[np.linalg.norm(p-t,axis=1).min() for t in actual_tips];index=int(np.argmin(distances))
            if distances[index]>2 or index in seen:raise ValueError('Contact does not corroborate a distinct observed head')
            seen.add(index);axis=(actual_tips[index]-center)/radius
            for identity in materials:
                q=np.asarray(lookup[identity]['support_points'],float)
                distances=np.linalg.norm(q-actual_tips[index],axis=1)
                if distances.min()>2:raise ValueError('Material path does not meet its contact foot')
                local=q[distances<=max(4.,8*np.sqrt(.5))]
                if len(local)<3:raise ValueError('Insufficient material-contact direction')
                _,_,v=np.linalg.svd(local-local.mean(0),full_matrices=False)
                if abs(float(v[0]@axis))>math.cos(math.radians(20)):
                    raise ValueError('Diameter shaft is not an independent material anchor')
