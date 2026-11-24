"""Certify untracked straight source paths for analysis, retaining exact owners."""
from copy import deepcopy
import math
import cv2
import numpy as np
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray,foreground

POLICY={'version':'open-section-source-linear/1.0','maximum_intervals':4096,
        'maximum_tracks':512,'maximum_source_points':2000000,'maximum_raster_work':64000000,
        'maximum_axis_deviation_px':math.sqrt(.5),'support':'own_8_connected_observed_pixels',
        'CAD_geometry_modified':False}


def observe(gray,graph):
    checked_gray(gray); before=object_sha(graph); image_sha=sha(gray)
    output=deepcopy(graph); lookup={p['id']:p for p in output['proposals']}
    if len(lookup)!=len(output['proposals']):raise ValueError('Unique source intervals required')
    audit={'source_points':0,'raster_work':0,'accepted':[],'rejected':{},'limits':[]}
    tracks=output.setdefault('tracks',[]); tracked={i for t in tracks for i in t['interval_ids']}
    if len(lookup)>POLICY['maximum_intervals']:audit['limits'].append('interval_budget')
    ink=foreground(gray)
    def reject(reason):audit['rejected'][reason]=audit['rejected'].get(reason,0)+1
    for p in output['proposals'] if not audit['limits'] else []:
        if p['id'] in tracked:continue
        if p.get('parent_kind') not in ('line','polyline'):reject('curved_or_unknown_parent');continue
        q=np.asarray(p['points_px'],float)
        if q.ndim!=2 or q.shape[1]!=2 or not len(q) or not np.isfinite(q).all() or not np.array_equal(q,np.rint(q)):
            raise ValueError('Finite integer source pixel centres required')
        q=q.astype(int);audit['source_points']+=len(q)
        if audit['source_points']>POLICY['maximum_source_points']:audit['limits'].append('source_point_budget');break
        if (q<0).any() or (q[:,0]>=gray.shape[1]).any() or (q[:,1]>=gray.shape[0]).any():
            raise ValueError('Source path outside pixels')
        if len(np.unique(q,axis=0))<3:reject('short_source');continue
        if not np.all(ink[q[:,1],q[:,0]]):reject('paper_in_source');continue
        fit=cv2.fitLine(q.astype(np.float32),cv2.DIST_L1,0,.01,.01).reshape(-1)
        axis=fit[:2].astype(float);center=fit[2:].astype(float);normal=np.array([-axis[1],axis[0]])
        deviation=float(np.max(abs((q-center)@normal)))
        if deviation>POLICY['maximum_axis_deviation_px']:reject('nonlinear_source');continue
        low,high=q.min(0),q.max(0)+1;audit['raster_work']+=int(np.prod(high-low))
        if audit['raster_work']>POLICY['maximum_raster_work']:audit['limits'].append('raster_budget');break
        walk=np.zeros(tuple((high-low)[::-1]),np.uint8);walk[q[:,1]-low[1],q[:,0]-low[0]]=1
        if cv2.connectedComponents(walk,connectivity=8)[0]!=2:reject('disconnected_own_source');continue
        t=(q-center)@axis;ends=q[[int(np.argmin(t)),int(np.argmax(t))]].astype(float)
        length=float(np.linalg.norm(ends[1]-ends[0]))
        if length==0:reject('degenerate_source');continue
        if len(tracks)>=POLICY['maximum_tracks']:audit['limits'].append('track_budget');break
        name='source-linear/'+object_sha(p['id'])[:24]
        if any(t['id']==name for t in tracks):raise ValueError('Source track identity collision')
        p['linear_context']=True
        tracks.append({'id':name,'interval_ids':[p['id']],'points':ends.tolist(),
                       'u':((ends[1]-ends[0])/length).tolist(),'length':length})
        audit['accepted'].append({'track_id':name,'source_interval_id':p['id'],'maximum_deviation_px':deviation})
    if audit['limits']:output=deepcopy(graph);output['search_incomplete']=True
    if before!=object_sha(graph) or image_sha!=sha(gray):raise ValueError('Source observation mutated input')
    return output,{'complete':not audit['limits'],'audit':audit,'policy':deepcopy(POLICY),
        'original_graph_sha256':before,'analysis_graph_sha256':object_sha(output),
        'source_immutable':True,'geometry_modified':False}
