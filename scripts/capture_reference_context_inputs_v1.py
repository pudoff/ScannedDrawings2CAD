"""Freeze target-free source inputs once; reuse the exact same observations in all arms."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, pin
from run_open_references_v1 import inputs
from run_topology_robust_v1 import read, write, object_hash

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'artifacts/reference-context-source-bank-v1'
CONTROLS={'path':'artifacts/reference-context-controls-v1/contract.json',
          'sha256':'9a8101dfd177c047e31c5c2fc9cc8ce25f16108d67b46f5655eb5699a83c8c2d'}
PREVIOUS={'path':'artifacts/open-reference-release-v1.json',
          'sha256':'a62592b191f4850bb99a46a9b7ba326aed5ef2357441fb09f92134371dba375f'}


def capture(split):
    cv2.setNumThreads(1); checked(ROOT,PREVIOUS)
    old=read(ROOT/'artifacts/open-reference-profile-v1/plan.json')
    for p in old['code']: checked(ROOT,p)
    if split=='train': jobs=[j for j in read(checked(ROOT,CONTROLS))['jobs'] if j['case']['split']=='train']
    else: jobs=old['jobs']
    old_index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/open-reference-profile-v1/results-index.json') if not r['repeat']}
    records=[]
    for job in jobs:
        views=[]
        for view in job['views']:
            gray,graph,base,heads,used,exact,calls=inputs(view)
            previous_features_exact=True
            if split=='previous':
                index=old_index[job['case']['id'],view['mode']]; folder=(ROOT/index['artifact']).parent
                previous_features_exact=read(folder/'features.json')=={k:v for k,v in graph.items() if k!='proposals'}
                assert previous_features_exact and len(base['dimension_chains'])==read(folder/'result.json')['previous_chains']
            folder=BASE/split/job['case']['id']/view['mode']; folder.mkdir(parents=True,exist_ok=False)
            write(folder/'graph.json',graph); write(folder/'heads.json',heads)
            write(folder/'base.json',{k:v for k,v in base.items() if not isinstance(v,np.ndarray)})
            Image.fromarray(base['pixel_roles']).save(folder/'roles.png')
            packet={n:pin(ROOT,(folder/file).relative_to(ROOT).as_posix()) for n,file in
                    [('graph','graph.json'),('heads','heads.json'),('base','base.json'),('roles','roles.png')]}
            views.append({'mode':view['mode'],'gray':view['gray'],'input_to_native':view['input_to_native'],
                          'bank':packet,'used_head_ids':[h['id'] for h in used],
                          'previous_features_exact':previous_features_exact,'original_real_features_exact':exact,
                          'capture_preparation':calls})
        records.append({**job,'views':views})
        if len(records)%20==0: print({'split':split,'captured_cases':len(records)},flush=True)
    write(BASE/(split+'-index.json'),{'previous':PREVIOUS,'controls':CONTROLS,'jobs':records,
          'all_previous_features_exact':True,'targets_opened':False,
          'generator':pin(ROOT,'scripts/capture_reference_context_inputs_v1.py')})
    print({'split':split,'cases':len(records),'views':sum(len(j['views']) for j in records)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','previous']);capture(p.parse_args().split)
