"""Train/development qualification, keeping the new strip-validation families unopened."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin,checked
from cad_recovery.open_reference_context_v2 import filter_context
from qualify_reference_context_v1 import arms as initial_arms
from run_reference_context_v1 import get_inputs,LABELS
from run_open_references_v1 import inputs,control_review
from cad_recovery.real_chain_metrics_v1 import head_metrics,chain_metrics
from run_topology_robust_v1 import read,write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/reference-context-refinement-qualification-v1'
CONTROLS={'path':'artifacts/reference-strip-controls-v1/contract.json','sha256':'a9341928be0ddd1429584a0e81b70fd746d18b6ce550657319f6433d7b15bc05'}


def arms(gray,graph,base,used):
    initial=initial_arms(gray,graph,base,used)
    return {'original_v1':initial['original_v1'],'basis_v2':initial['basis_v2'],
            'context_v1':initial['context_v2'],'context_v2':filter_context(gray,graph,base,initial['basis_v2'])}


def qualify():
    cv2.setNumThreads(1);cv2.setRNGSeed(0);BASE.mkdir(exist_ok=False);jobs=[];rows=[]
    for job in read(checked(ROOT,CONTROLS))['jobs']:
        if job['case']['split']!='train':continue
        views=[]
        for view in job['views']:
            gray,graph,base,heads,used,_,calls=inputs(view)
            folder=BASE/'bank'/job['case']['id']/view['mode'];folder.mkdir(parents=True)
            write(folder/'graph.json',graph);write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
            bank={key:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for key,file in
                  [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]}
            views.append({**view,'bank':bank,'used_head_ids':[h['id'] for h in used],'capture_preparation':calls})
        jobs.append({**job,'views':views})
    old=read(ROOT/'artifacts/reference-context-profile-v1/plan.json')
    failed={r['case'] for r in read(ROOT/'artifacts/open-reference-profile-v1/report.json')['false_new_chains_on_negatives']}
    development=[{**j,'case':{**j['case'],'split':'development'}} for j in old['jobs']
                 if j['case']['id'] in failed|{'train-3-left','train-2-upper'}]
    for job in jobs+development:
        for view in job['views']:
            gray,graph,base,heads,used,_=get_inputs(view);results=arms(gray,graph,base,used);metrics={}
            review=control_review(job['case']) if job['case'].get('labels',{}).get('feet_px') else next(
                (r for r in read(checked(ROOT,LABELS))['records'] if r['id']==job['case']['id']),None)
            hm=head_metrics(review,heads,view['input_to_native']) if review else {}
            for arm,result in results.items():
                if job['case'].get('labels',{}).get('expected_dimension_chains')==0:
                    metrics[arm]={'tp':0,'fp':len(result['open_reference_chains']),'fn':0}
                elif review:
                    m=chain_metrics(review,heads,view['input_to_native'],result,graph,{'contacts':{},'eligible_boundaries':[]},hm)
                    metrics[arm]={k:m[k] for k in ('tp','fp','fn')}
            rows.append({'case':job['case']['id'],'mode':view['mode'],'split':job['case']['split'],'metrics':metrics,
              'counts':{a:len(r['open_reference_chains']) for a,r in results.items()},
              'decisions':results['context_v2']['part_structure_decisions']})
    write(BASE/'train-index.json',{'jobs':jobs,'controls':CONTROLS,'validation_predictions_opened':False})
    write(BASE/'report.json',{'rows':rows,'validation_predictions_opened':False,'code':[pin(ROOT,n) for n in
        ('src/cad_recovery/open_reference_context_v2.py','scripts/qualify_reference_context_v2.py','tests/test_reference_context_v2.py')]})
    for split in ('train','development'):
        selected=[r for r in rows if r['split']==split]
        print({'split':split,'metrics':{a:{k:sum(r['metrics'].get(a,{}).get(k,0) for r in selected)
               for k in ('tp','fp','fn')} for a in ('original_v1','basis_v2','context_v1','context_v2')}},flush=True)


if __name__=='__main__':qualify()
