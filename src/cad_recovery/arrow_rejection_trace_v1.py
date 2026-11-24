"""Observe rejection gates of the pinned detector without copying its algorithm.

Only explicit continue statements in _line_arrows are observed. Frame locals
are read, never changed; the previous Python trace hook is restored on exit.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import inspect
import math
import sys

import numpy as np

from .arrow_detection import _line_arrows, MARGIN
from .family_catalog_v2 import require

GATES = ['wide_run_length','fit_samples','fit_inlier_fraction','taper_fit',
         'taper_height','asymmetric_profile','head_length_or_aspect',
         'missing_abrupt_base','missing_connected_shaft','tip_outside_image']
POLICY = {'version':'pinned-arrow-rejection-trace/1.0','maximum_rejection_events':50000,
          'maximum_line_calls':30000,'nearby_run_distance_px':90,
          'supporting_axis_distance_px':8,'supporting_axis_angle_deg':20,
          'detector_thresholds_changed':False,'manual_references_changed':False,
          'production_admitted':False,'physical_geometry_approved':False}


def gate_map():
    source,first = inspect.getsourcelines(_line_arrows)
    tree = ast.parse(''.join(source))
    nodes = sorted((n for n in ast.walk(tree) if isinstance(n,ast.Continue)),key=lambda n:n.lineno)
    require(len(nodes)==len(GATES),'Pinned detector rejection structure changed')
    parents = {child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    result = {}
    for name,node in zip(GATES,nodes):
        branch = parents[node]
        while not isinstance(branch,ast.If):branch=parents[branch]
        result[first+node.lineno-1] = {'reason':name,'source_line':first+node.lineno-1,
                                     'predicate':ast.unparse(branch.test)}
    return result


def finite_scalar(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,float,np.integer,np.floating)):
        return float(value) if math.isfinite(value) else None
    return None


class RejectionTrace:
    def __init__(self):
        self.gates = gate_map()
        self.rows = []
        self.rejections = 0
        self.active = {}

    def trace(self,frame,event,arg):
        if frame.f_code is not _line_arrows.__code__:return None
        if event=='call':
            require(len(self.rows)<POLICY['maximum_line_calls'],'Arrow trace call budget exceeded')
            line=frame.f_locals['line']
            row={'geometry_id':line['id'],'points_px':deepcopy(line['points_px']),'rejections':[]}
            self.rows.append(row);self.active[id(frame)]=row
        row=self.active.get(id(frame))
        if row is None:return self.trace
        if event=='line' and frame.f_lineno in self.gates:
            require(self.rejections<POLICY['maximum_rejection_events'],'Arrow rejection trace budget exceeded')
            self.rejections+=1
            values=frame.f_locals
            entry={**self.gates[frame.f_lineno],
                   'run_indices':[int(values['begin']),int(values['end'])],
                   'run_length_samples':int(values['end']-values['begin']),
                   'measured_shaft_width_px':finite_scalar(values.get('stroke')),
                   'wide_run_threshold':finite_scalar(values.get('threshold'))}
            reason=entry['reason']
            if reason not in ('wide_run_length','fit_samples'):
                entry['fit_inlier_fraction']=float(np.mean(values['keep']))
            if reason in GATES[3:]:
                for key in ('slope','intercept','r_squared'):
                    entry[key]=finite_scalar(values.get(key))
            if reason in GATES[5:]:entry['center_slope']=finite_scalar(values.get('center_slope'))
            if reason in GATES[6:]:
                for key in ('arrow_length','base_width','base_index','tip_index'):
                    entry[key]=finite_scalar(values.get(key))
            if reason in GATES[8:]:entry['shaft_support']=finite_scalar(values.get('shaft'))
            row['rejections'].append(entry)
        if event=='return':
            values=frame.f_locals
            row['accepted_before_text_and_dedup']=len(arg) if isinstance(arg,list) else 0
            row['measured_shaft_width_px']=finite_scalar(values.get('stroke'))
            row['wide_run_threshold']=finite_scalar(values.get('threshold'))
            if row['accepted_before_text_and_dedup']:
                row['terminal_state']='has_candidate'
            elif len(values.get('supported',[]))<20:
                row['terminal_state']='insufficient_supported_profile'
            elif not row['rejections']:
                row['terminal_state']='no_wide_run'
            else:row['terminal_state']='all_wide_runs_rejected'
            self.active.pop(id(frame),None)
        return self.trace

    @contextmanager
    def observe(self):
        previous=sys.gettrace()
        sys.settrace(self.trace)
        try:yield self
        finally:
            sys.settrace(previous)
            self.active.clear()


def nearby_rejections(reference,rows,matrix):
    """Assigned manual tips locate diagnostic events, never alter detector inputs."""
    matrix=np.asarray(matrix,float)
    output=[]
    for arrow in reference['arrows']:
        tip=np.asarray(arrow['tip'],float)
        direction=np.asarray(arrow['direction'],float);direction/=np.linalg.norm(direction)
        support=[]
        for row in rows:
            points=np.asarray(row['points_px'],float)
            homogeneous=np.c_[points,np.ones(2)]@matrix.T
            line=homogeneous[:,:2]/homogeneous[:,2:]
            delta=line[1]-line[0];length=float(np.linalg.norm(delta));unit=delta/length
            angle=math.degrees(math.acos(float(np.clip(abs(unit@direction),0,1))))
            projection=float((tip-line[0])@unit)
            perpendicular=float(np.linalg.norm(tip-(line[0]+projection*unit)))
            if (angle>POLICY['supporting_axis_angle_deg'] or perpendicular>POLICY['supporting_axis_distance_px']
                    or not -MARGIN<=projection<=length+MARGIN):continue
            # Current cohort maps are Euclidean translations/rotations; reject scale/shear rather than mislabel distances.
            require(np.allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-6), 'Trace requires rigid pixel transform')
            events=[]
            for event in row['rejections']:
                left,right=np.asarray(event['run_indices'])-MARGIN
                distance=max(float(left-projection),float(projection-right),0)
                if distance<=POLICY['nearby_run_distance_px']:
                    events.append({**event,'distance_to_run_px':distance})
            support.append({'geometry_id':row['geometry_id'],'axis_distance_px':perpendicular,
                'axis_angle_error_deg':angle,'terminal_state':row['terminal_state'],
                'accepted_before_text_and_dedup':row['accepted_before_text_and_dedup'],
                'nearby_rejections':events})
        output.append({'id':arrow['id'],'supporting_straight_axes':support,
                       'no_compatible_straight_axis':not support,
                       'causal_scope':'Nearby gates on compatible straight axes; not a unique rejection cause when several runs/axes are present.'})
    return output
