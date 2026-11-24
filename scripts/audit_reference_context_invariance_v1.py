"""Train/real representation checks; no fresh-validation model evaluation."""
import ast
from copy import deepcopy
from pathlib import Path
from cad_recovery.dataset_layout_v1 import pin
from cad_recovery.open_reference_chains_v2 import extend
from cad_recovery.open_reference_context_v1 import filter_context
from qualify_reference_context_v1 import load
from run_topology_robust_v1 import read,write,object_hash

ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'artifacts/reference-context-source-bank-v1'


def signature(packet):
    return sorted((tuple(sorted((h,tuple(round(float(v),8) for v in p)) for h,p in zip(c['head_ids'],c['feet_px']))))
                  for c in packet['open_reference_chains'])


def audit():
    jobs=read(BANK/'train-index.json')['jobs']
    jobs += [j for j in read(BANK/'previous-index.json')['jobs'] if j['case']['id']=='train-3-left']
    rows=[]
    for job in jobs:
        for view in job['views']:
            gray,graph,base,heads,used=load(view);snap=object_hash([graph,heads,base['dimension_chains']])
            reference=extend(gray,graph,base,used); guarded=filter_context(gray,graph,base,reference)
            for mode in ('single_extension_basis','all_track_bases','source_order_and_directions'):
                changed=deepcopy(graph); changed_heads=list(used)
                if mode=='single_extension_basis' and reference['open_reference_chains']:
                    key=reference['open_reference_chains'][0]['local_reference_evidence']['references'][0]['extension_track']
                    track=next(t for t in changed['tracks'] if t['id']==key)
                    track['points']=track['points'][::-1];track['u']=[-u for u in track['u']]
                elif mode=='all_track_bases':
                    for t in changed['tracks']:t['points']=t['points'][::-1];t['u']=[-u for u in t['u']]
                elif mode=='source_order_and_directions':
                    changed['tracks'].reverse();changed['proposals'].reverse();changed_heads.reverse()
                    for p in changed['proposals']:p['points_px'].reverse()
                    for t in changed['tracks']:t['interval_ids'].reverse()
                result=extend(gray,changed,base,changed_heads);context=filter_context(gray,changed,base,result)
                rows.append({'case':job['case']['id'],'mode':view['mode'],'probe':mode,
                             'basis_invariant':signature(result)==signature(reference),
                             'context_invariant':signature(context)==signature(guarded),
                             'source_unchanged':snap==object_hash([graph,heads,base['dimension_chains']]),
                             'incomplete':result['search_incomplete'] or context['search_incomplete']})
    trees=[ast.parse((ROOT/f'src/cad_recovery/open_reference_chains_v{i}.py').read_text(encoding='utf-8')) for i in (1,2)]
    unchanged={name:ast.dump(next(n for n in trees[0].body if getattr(n,'name',None)==name),include_attributes=False)==
                    ast.dump(next(n for n in trees[1].body if getattr(n,'name',None)==name),include_attributes=False)
               for name in ('LocalReferences','contacts')}
    write(ROOT/'artifacts/reference-context-qualification-v1/invariance.json',{'rows':rows,'unchanged_kernel_methods':unchanged,
          'all_invariant':all(r['basis_invariant'] and r['context_invariant'] and r['source_unchanged'] and not r['incomplete'] for r in rows),
          'fresh_validation_used':False,'code':[pin(ROOT,p) for p in ('src/cad_recovery/open_reference_chains_v2.py','src/cad_recovery/open_reference_context_v1.py')]})
    print({'probes':len(rows),'all_invariant':all(r['basis_invariant'] and r['context_invariant'] for r in rows),'kernel_unchanged':unchanged},flush=True)


if __name__=='__main__':audit()
