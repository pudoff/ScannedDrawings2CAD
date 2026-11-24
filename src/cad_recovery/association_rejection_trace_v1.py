"""Read actual v3 reject branches without copying or changing its predicates."""
import ast
from collections import Counter
from contextlib import contextmanager
import inspect
import math
import sys
import numpy as np
from . import dimension_association_v3 as v3

POLICY={'version':'association-rejection-trace/1.0','maximum_events':12000,
        'maximum_calls':10000,'sampled_events_per_branch':16,'decisions_changed':False}
FUNCTIONS=(v3._linear,v3._text_placement,v3._anchor_options)

def _small(value):
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,float,np.integer,np.floating)):return float(value) if math.isfinite(value) else None
    if isinstance(value,str):return value
    if isinstance(value,dict):
        return {k:_small(value[k]) for k in ('id','u','center','width','height','tol','tip','points') if k in value}
    if isinstance(value,np.ndarray):return value.tolist() if value.size<=8 else {'elements':int(value.size)}
    if isinstance(value,(list,tuple)):
        return [_small(v) for v in value] if len(value)<=4 else {'count':len(value)}
    if value is None:return None
    return {'type':type(value).__name__}

def gate_map(function):
    source,first=inspect.getsourcelines(function);tree=ast.parse(''.join(source))
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)};gates={}
    for node in ast.walk(tree):
        reject=isinstance(node,ast.Continue) or isinstance(node,ast.Return) and isinstance(node.value,ast.Constant) and node.value.value is None
        if not reject:continue
        parent=parents[node]
        while parent in parents and not isinstance(parent,ast.If):parent=parents[parent]
        if not isinstance(parent,ast.If):raise ValueError('Reject branch needs an explicit predicate')
        predicate=ast.unparse(parent.test);names=sorted({n.id for n in ast.walk(parent.test) if isinstance(n,ast.Name)})
        line=first+node.lineno-1
        gates[line]={'function':function.__name__,'source_line':line,'predicate':predicate,
                     'branch_id':function.__name__+':'+str(line),'names':names}
    return gates

class RejectionTrace:
    def __init__(self):
        self.maps={f.__code__:gate_map(f) for f in FUNCTIONS};self.calls=[];self.events=[];self.active={}
        self.counts=Counter();self.samples=Counter();self.events_seen=0;self.dropped_events=0;self.dropped_calls=0

    def trace(self,frame,event,arg):
        if frame.f_code not in self.maps:return None
        if event=='call':
            if len(self.calls)>=POLICY['maximum_calls']:self.dropped_calls+=1;return None
            caller=frame.f_back;parent=None
            while caller:
                if id(caller) in self.active:parent=self.active[id(caller)]['id'];break
                caller=caller.f_back
            local=frame.f_locals
            row={'id':'call-'+str(len(self.calls)+1),'parent_call_id':parent,'function':frame.f_code.co_name,
                 'text_id':local.get('text',{}).get('id'),'baseline_id':local.get('baseline',{}).get('id'),
                 'arrow_ids':[local[n]['id'] for n in ('first','second') if n in local]}
            self.calls.append(row);self.active[id(frame)]=row
        row=self.active.get(id(frame))
        if row is None:return self.trace
        if event=='exception':row['exception_type']=arg[0].__name__
        if event=='line' and frame.f_lineno in self.maps[frame.f_code]:
            gate=self.maps[frame.f_code][frame.f_lineno];self.events_seen+=1;self.counts[gate['branch_id']]+=1
            if len(self.events)<POLICY['maximum_events'] and self.samples[gate['branch_id']]<POLICY['sampled_events_per_branch']:
                self.events.append({k:v for k,v in gate.items() if k!='names'}|{
                    'call_id':row['id'],'text_id':row['text_id'],'baseline_id':row['baseline_id'],'arrow_ids':row['arrow_ids'],
                    'witness':{k:_small(frame.f_locals[k]) for k in gate['names'] if k in frame.f_locals}})
                self.samples[gate['branch_id']]+=1
            else:self.dropped_events+=1
        if event=='return':
            if row.get('exception_type') and arg is None:row['result']='exception_exit_not_a_geometric_rejection'
            elif frame.f_code is v3._text_placement.__code__:row['result']='rejected' if arg is None else 'placement_supported'
            elif isinstance(arg,tuple):
                row['alternatives_count']=len(arg[0]);row['truncated']=bool(arg[1])
            else:row['result']='exception_or_nonstandard_return'
            self.active.pop(id(frame),None)
        return self.trace

    @contextmanager
    def observe(self):
        previous=sys.gettrace();sys.settrace(self.trace)
        try:yield self
        finally:sys.settrace(previous);self.active.clear()

    def report(self):
        return {'policy':POLICY,'calls':self.calls,'events':self.events,'branch_counts':dict(self.counts),
                'events_seen':self.events_seen,'unsampled_events':self.dropped_events,'dropped_calls':self.dropped_calls,
                'call_inventory_complete':self.dropped_calls==0,'decisions_changed':False,
                'sampling':'counts cover captured calls; bounded witness samples per branch; unsampled events are explicit'}
