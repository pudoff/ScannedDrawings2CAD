"""Source-owned analysis chords for unresolved open paths, never CAD objects."""
from copy import deepcopy
import cv2
import numpy as np
from .part_admission_v1 import object_sha
from .stroke_curve_control_v1 import distance_to_polyline


def probes(result,geometry,*,maximum_segments=4096):
    lookup={c['id']:c for c in geometry};output=[];audit={'complete':True,'visited_segments':0,
        'analysis_lines':0,'rejected_segments':0,'max_error_px':2.,'p95_error_px':1.25,'CAD_geometry_modified':False}
    for obj in result['objects']:
        if obj['kind']!='polyline' or obj.get('closed'):
            output.append(obj);continue
        for member in obj['source_members']:
            c=lookup[member['candidate_id']];start,end=member['source_index_interval'];origin=c['source_index_interval'][0]
            q=np.asarray(c['support_points'][start-origin:end-origin+1],float)
            if len(q)<2:continue
            vertices=cv2.approxPolyDP(q.astype(np.float32),1.25,False).reshape(-1,2)
            indices=[0];position=0
            for vertex in vertices[1:-1]:
                matches=np.flatnonzero(np.all(q[position+1:]==vertex,axis=1))
                if len(matches):position+=int(matches[0])+1;indices.append(position)
            indices.append(len(q)-1)
            for lo,hi in zip(indices,indices[1:]):
                audit['visited_segments']+=1
                if audit['visited_segments']>maximum_segments:audit['complete']=False;return result['objects'],audit
                p=q[lo:hi+1];ends=p[[0,-1]];errors=distance_to_polyline(p,ends)
                if np.linalg.norm(ends[1]-ends[0])<6 or errors.max()>2 or np.percentile(errors,95)>1.25:
                    audit['rejected_segments']+=1;continue
                m=deepcopy(member);m['source_index_interval']=[start+lo,start+hi]
                if member.get('graph_interval_ranges'):
                    m['graph_interval_ranges']=[{'id':r['id'],'source_index_interval':[
                        max(start+lo,r['source_index_interval'][0]),min(start+hi,r['source_index_interval'][1])]}
                        for r in member['graph_interval_ranges'] if max(start+lo,r['source_index_interval'][0])<min(start+hi,r['source_index_interval'][1])]
                    m['interval_ids']=[r['id'] for r in m['graph_interval_ranges']]
                output.append({**obj,'id':'local-analysis-line/'+object_sha(m)[:24],'kind':'line','points_px':ends.tolist(),
                    'support_points':p.tolist(),'source_members':[m],'analysis_probe_only':True,'parent_unresolved_object':obj['id']})
                audit['analysis_lines']+=1
    return output,audit
