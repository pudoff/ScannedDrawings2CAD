"""Train-only ablation; do not open fresh validation targets during qualification."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.open_reference_chains_v1 import extend as original
from cad_recovery.open_reference_chains_v2 import extend as basis
from cad_recovery.open_reference_context_v1 import filter_context
from cad_recovery.real_chain_metrics_v1 import head_metrics, chain_metrics
from cad_recovery.stroke_roles_v1 import foreground
from cad_recovery.open_reference_chains_v1 import contacts
from run_open_references_v1 import control_review
from run_topology_robust_v1 import read, write, object_hash

ROOT=Path(__file__).resolve().parents[1]; BANK=ROOT/'artifacts/reference-context-source-bank-v1'


def load(view):
    gray=np.array(Image.open(checked(ROOT,view['gray'])).convert('L')); cache=view['bank']
    graph=read(checked(ROOT,cache['graph'])); heads=read(checked(ROOT,cache['heads'])); base=read(checked(ROOT,cache['base']))
    base['pixel_roles']=np.array(Image.open(checked(ROOT,cache['roles'])))
    allowed=set(view['used_head_ids']); used=[h for h in heads if h['id'] in allowed]
    return gray,graph,base,heads,used


def arms(gray,graph,base,used):
    old=original(gray,graph,base,used); corrected=basis(gray,graph,base,used)
    return {'original_v1':old,'basis_v2':corrected,'context_v2':filter_context(gray,graph,base,corrected)}


def qualify(output):
    cv2.setNumThreads(1); rows=[]
    for job in read(BANK/'train-index.json')['jobs']:
        for view in job['views']:
            gray,graph,base,heads,used=load(view); before=object_hash([graph,heads,base['dimension_chains']])
            results=arms(gray,graph,base,used)
            assert before==object_hash([graph,heads,base['dimension_chains']])
            review=control_review(job['case']); hm=head_metrics(review,heads,view['input_to_native'])
            trace={'contacts':{k:[h['id'] for h in v] for k,v in contacts(graph['tracks'],used,foreground(gray)).items()},'eligible_boundaries':[]}
            metrics={}
            for name,packet in results.items():
                if not job['case']['labels']['expected_dimension_chains']:
                    metrics[name]={'tp':0,'fp':len(packet['open_reference_chains']),'fn':0}
                else:
                    score=chain_metrics(review,heads,view['input_to_native'],{**packet,'dimension_chains':packet['open_reference_chains']},graph,trace,hm)
                    metrics[name]={k:score[k] for k in ('tp','fp','fn')}
            rows.append({'case':job['case']['id'],'mode':view['mode'],'metrics':metrics,
              'incomplete':any(p['search_incomplete'] for p in results.values()),
              'context_decisions':results['context_v2']['reference_context_decisions']})
    write(ROOT/output,{'train_only':True,'rows':rows,'code':[pin(ROOT,p) for p in
          ('src/cad_recovery/open_reference_chains_v2.py','src/cad_recovery/open_reference_context_v1.py')]})
    print({a:{k:sum(r['metrics'][a][k] for r in rows) for k in ('tp','fp','fn')} for a in ('original_v1','basis_v2','context_v2')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);qualify(p.parse_args().output)
