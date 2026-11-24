"""Bind legacy working-raster observations to the frozen contour-veto study.

This is a diagnostic adapter, not a new CV producer or a production admission.
Incomplete historical searches remain incomplete; source owners are rebuilt
from the checked observed graph, never inferred from a target role map.
"""
import json
from pathlib import Path
import numpy as np
from .cad_object_inputs_v2 import load_observations
from .dataset_layout_v1 import checked
from .supported_role_overlay_v1 import source_ownership
from .stroke_roles_v1 import foreground
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


def load(root,folder):
    packet,protected,refs,extra=load_observations(root,folder)
    graph=json.loads(checked(root,refs['graph.json']).read_text(encoding='utf8'))
    heads=json.loads(checked(root,refs['heads.json']).read_text(encoding='utf8')) if 'heads.json' in refs else []
    gray=packet['gray'];before=object_sha([graph,heads]);pixels=sha(gray)
    ownership,work=source_ownership(foreground(gray),graph['proposals'])
    if ownership is None:
        owner=np.zeros_like(gray,np.uint16);ambiguous=np.zeros_like(gray)
    else:
        nearest=ownership['owner'];owner=np.maximum(nearest,0).astype(np.uint16)
        ambiguous=np.where(nearest<0,255,0).astype(np.uint8)
        for number,interval in enumerate(graph['proposals'],1):
            if len(interval['points_px'])!=1:continue
            x,y=np.rint(interval['points_px'][0]).astype(int)
            if owner[y,x] not in (0,number):ambiguous[y,x]=255;owner[y,x]=0
            elif not ambiguous[y,x]:owner[y,x]=number
    if owner.max(initial=0)>len(graph['proposals']) or np.any((ambiguous>0)&(owner!=0)):
        raise ValueError('Invalid source ownership')
    complete=bool(packet['summary'].get('search_complete') is True and
        graph.get('search_incomplete') is False and not work['incomplete'])
    if before!=object_sha([graph,heads]) or pixels!=sha(gray):raise ValueError('Observation adapter mutated its source')
    result={**packet,'graph':graph,'heads':heads,'source_owner':owner,'ambiguous_owner_mask':ambiguous,
        'complete':complete,'input_sha256':pixels,'files':{'roles':refs['roles.png']}}
    return result,protected,refs,{'source_owner_rebuilt_from_checked_graph':True,'ownership':work,
        'historical_search_complete':packet['summary'].get('search_complete'),
        'graph_search_incomplete':graph.get('search_incomplete'),'working_raster_is_full_raw_original':False,
        'new_CV_extraction':False,'semantic_ownership_confirmed':False,**extra}
