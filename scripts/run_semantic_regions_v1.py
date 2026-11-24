"""Prepare/freeze first, then train-ROI diagnostics and two validation repeats."""
from pathlib import Path
import sys
from cad_recovery.dataset_layout_v1 import encoded,read,pin
from cad_recovery.family_catalog_v2 import require
from cad_recovery.semantic_region_experiment_v1 import prepare,run_worker,evaluate

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]; folder=root/'artifacts/semantic-region-experiment-v1'
    if sys.argv[1]=='prepare':
        spec=prepare(root,folder); (folder/'spec.json').write_bytes(encoded(spec))
        print({'spec':pin(root,(folder/'spec.json').relative_to(root).as_posix()),'pixels':spec['total_pixels']})
    elif sys.argv[1]=='run':
        spec=read(folder/'spec.json'); resources=[]
        for task in ('classify_roi','localize_page'):
            for repeat in (1,2):
                output=folder/f'{task}-{repeat}.json'; require(not output.exists(),'Choose new worker output')
                resources.append({'task':task,'repeat':repeat,**run_worker(root,spec,output,task)})
            a=read(folder/f'{task}-1.json'); b=read(folder/f'{task}-2.json')
            require([(r['id'],r['prediction']) for r in a['rows']]==[(r['id'],r['prediction']) for r in b['rows']], 'CV repeat differs')
        result=evaluate(root,spec,read(folder/'classify_roi-1.json'),read(folder/'localize_page-1.json'))
        result['resources']=resources; result['two_fresh_process_repeats_equal']=True
        (folder/'report.json').write_bytes(encoded(result)); print({k:v for k,v in result.items() if k not in ('validation','train_roi_classification')})
    else:raise ValueError('Use prepare or run')
