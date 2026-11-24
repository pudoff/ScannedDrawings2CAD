"""Read-only audit of actual list filters as well as explicit reject branches.

Filter predicates are compiled from pinned v3 source, checked for pure calls,
and replayed against frame inputs. Their kept IDs must equal the producer's
actual list. No producer local, budget or decision is changed.
"""
import ast
from collections import Counter
import inspect
from .association_rejection_trace_v1 import RejectionTrace as OriginalTrace,_small,FUNCTIONS
from . import dimension_association_v3 as v3

POLICY={'version':'association-rejection-trace/2.0','maximum_filter_snapshots':4000,
        'sampled_filter_rejections_per_predicate':24,'decisions_changed':False,
        'scope':'post-run audit only; no OCR, detector or threshold selection'}

def _compile(node):
    for item in ast.walk(node):
        if isinstance(item,ast.Call):
            name=ast.unparse(item.func)
            if name not in {'abs','float','max','min','any','np.dot','_perp','_segment_distance'}:
                raise ValueError('Unapproved side effect in diagnostic predicate: '+name)
    return compile(ast.fix_missing_locations(ast.Expression(body=node)),'<pinned-v3-filter>','eval')

def filter_map():
    source,first=inspect.getsourcelines(v3._linear);tree=ast.parse(''.join(source))
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)};result={}
    for node in ast.walk(tree):
        if not(isinstance(node,ast.Assign) and isinstance(node.value,ast.ListComp)
               and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'near_lines','members','relevant'}):continue
        generator=node.value.generators[0]
        if not(isinstance(generator.target,ast.Name) and isinstance(generator.iter,ast.Call)
               and isinstance(generator.iter.func,ast.Name) and generator.iter.func.id=='_budgeted'
               and isinstance(generator.iter.args[0],ast.Name)):raise ValueError('Pinned filter structure changed')
        parent=parents[node]
        block=next(value for _,value in ast.iter_fields(parent) if isinstance(value,list) and node in value)
        next_line=first+block[block.index(node)+1].lineno-1
        predicates=[]
        for condition in generator.ifs:
            predicates += condition.values if isinstance(condition,ast.BoolOp) and isinstance(condition.op,ast.And) else [condition]
        result[next_line]={'name':node.targets[0].id,'variable':generator.target.id,
            'input_name':generator.iter.args[0].id,'source_line':first+node.lineno-1,
            'predicates':[{'code':_compile(p),'source':ast.unparse(p),
                           'names':sorted({n.id for n in ast.walk(p) if isinstance(n,ast.Name)})} for p in predicates]}
    if {v['name'] for v in result.values()}!={'near_lines','members','relevant'}:raise ValueError('Incomplete pinned filter map')
    return result

def _context(frame):
    current=frame;baseline=None;arrows=[]
    while current:
        local=current.f_locals
        if baseline is None and isinstance(local.get('baseline'),dict):baseline=local['baseline'].get('id')
        if not arrows and all(isinstance(local.get(k),dict) and 'id'in local[k] for k in ('first','second')):
            arrows=[local[k]['id'] for k in ('first','second')]
        current=current.f_back
    return baseline,arrows

class RejectionTrace(OriginalTrace):
    def __init__(self):
        super().__init__();self.filters=filter_map();self.snapshots=[];self.filter_counts=Counter();self.filter_events=[]
        self.filter_samples=Counter();self.dropped_snapshots=0
        self.actions={}
        for function in FUNCTIONS:
            source,first=inspect.getsourcelines(function)
            for node in ast.walk(ast.parse(''.join(source))):
                if isinstance(node,ast.Continue):self.actions[first+node.lineno-1]='skipped_loop_item'
                elif isinstance(node,ast.Return) and isinstance(node.value,ast.Constant) and node.value.value is None:
                    self.actions[first+node.lineno-1]='function_returned_none'

    def trace(self,frame,event,arg):
        if frame.f_code not in self.maps:return None
        old_len=len(self.events);result=super().trace(frame,event,arg)
        row=self.active.get(id(frame))
        baseline,arrows=_context(frame)
        if event=='call' and row and frame.f_code is not v3._linear.__code__:
            row['baseline_id']=baseline
        if len(self.events)>old_len:
            item=self.events[-1];item['baseline_id']=baseline;item['arrow_ids']=arrows
            item['control_flow_action']=self.actions.get(item['source_line'],'unknown')
        if event=='line' and frame.f_code is v3._linear.__code__ and frame.f_lineno in self.filters:
            self._filter(frame,self.filters[frame.f_lineno],row)
        return result

    def _filter(self,frame,descriptor,row):
        if len(self.snapshots)>=POLICY['maximum_filter_snapshots']:self.dropped_snapshots+=1;return
        local=frame.f_locals;context={**frame.f_globals,**local};name=descriptor['name'];observed=local[name]
        expected=[];rejects=Counter();baseline,_=_context(frame)
        for candidate in local[descriptor['input_name']]:
            scope=context|{descriptor['variable']:candidate};failed=None
            for index,predicate in enumerate(descriptor['predicates']):
                if not bool(eval(predicate['code'],scope,scope)):
                    failed=index;break
            if failed is None:expected.append(candidate['id']);continue
            predicate=descriptor['predicates'][failed];key=name+':'+str(failed)
            rejects[key]+=1;self.filter_counts[key]+=1
            if self.filter_samples[key]<POLICY['sampled_filter_rejections_per_predicate']:
                self.filter_samples[key]+=1
                entry={'stage':name,'predicate_index':failed,'predicate':predicate['source'],
                    'control_flow_action':'filtered_observation','source_line':descriptor['source_line'],
                    'call_id':row['id'] if row else None,'text_id':local['text']['id'],'baseline_id':baseline,
                    'candidate_id':candidate['id'],
                    'witness':{k:_small(scope[k]) for k in predicate['names'] if k in scope}}
                if name=='relevant':
                    entry['tip_to_member_min_distance_px']=min((v3._segment_distance(candidate['tip'],m) for m in local['members']),default=None)
                    entry['text_tolerance_px']=float(local['text']['tol'])
                self.filter_events.append(entry)
        actual=[item['id'] for item in observed]
        if actual!=expected:raise ValueError('Diagnostic filter replay disagrees with actual producer list')
        self.snapshots.append({'stage':name,'text_id':local['text']['id'],'baseline_id':baseline,
            'source_line':descriptor['source_line'],'actual_kept_ids':actual,'predicate_replay_kept_ids':expected,
            'rejection_counts':dict(rejects),'input_count':len(local[descriptor['input_name']])})

    def report(self):
        return super().report()|{'audit_policy':POLICY,'filter_snapshots':self.snapshots,
            'filter_rejections':self.filter_events,'filter_rejection_counts':dict(self.filter_counts),
            'dropped_filter_snapshots':self.dropped_snapshots,
            'filter_inventory_complete':self.dropped_snapshots==0,'inference_or_parameters_changed':False}
