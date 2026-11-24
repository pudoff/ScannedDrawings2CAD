"""Pre-freeze qualification on two already-viewed large regression sources."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.streamed_local_heads_v1 import search
from cad_recovery.head_support_ownership_v1 import apply
from run_local_width_heads_v1 import read, write

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    plan=read(checked(ROOT,{'path':'artifacts/head-support-ownership-v1/plan.json',
        'sha256':'0c9f4a520eb92fb757b4345b3e829ba554b6a2f56b186d6270c85b42456a39d3'}))
    index={(r['case'],r['mode']):r for r in read(ROOT/'artifacts/head-support-ownership-v1/results-index.json') if not r['repeat']}
    outputs=[]
    for case in ('historical-legacy-val149-upper','historical-legacy-val162-left'):
        job=next(j for j in plan['jobs'] if j['case']['id']==case);view=job['views'][0]
        old=read(ROOT/index[case,view['mode']]['artifact']);gray=np.array(Image.open(checked(ROOT,view['gray'])).convert('L'))
        started=time.perf_counter();raw=search(gray,baseline=old['v1_search']['candidates'])
        guarded=apply(gray,raw,baseline=old['v1_search']['candidates'])
        row={'case':case,'source':view['gray'],'previous_result':pin(ROOT,index[case,view['mode']]['artifact']),
            'seconds':time.perf_counter()-started,'raw':raw,'guarded':guarded}
        outputs.append(row)
        print(json.dumps({'case':case,'samples':raw['analysis']['profile_samples'],
            'complete':not raw['search_incomplete'],'generated':raw['analysis']['raw_regions_generated'],
            'processed':raw['analysis']['raw_regions_processed'],'statistics':raw['analysis']['profile_streaming'],
            'seconds':row['seconds']}),flush=True)
    write(ROOT/'artifacts/streamed-head-viewed-qualification-v1.json',{
        'scope':'viewed development only; new controls not used','new_raw_search_calls':2,
        'new_ownership_calls':2,'parameters_changed':False,'rows':outputs})
