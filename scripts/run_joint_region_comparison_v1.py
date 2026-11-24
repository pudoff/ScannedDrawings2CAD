"""Freeze, execute guarded workers, or replay saved observations."""
import argparse
from pathlib import Path
from cad_recovery.dataset_layout_v1 import encoded, pin, read
from cad_recovery.joint_region_comparison_v1 import FOLDER, prepare, authenticate, evaluate
from cad_recovery.hard_raster_runs_v1 import execute

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','evaluate']);p.add_argument('--sha256');a=p.parse_args()
    if a.action=='prepare':
        config=prepare(ROOT);print('prepared',config['roi_counts'],pin(ROOT,FOLDER+'/protocol.json'));return
    if not a.sha256:p.error('--sha256 must be the protocol hash recorded before inference')
    config=authenticate(ROOT,{'path':FOLDER+'/protocol.json','sha256':a.sha256})
    if a.action=='run':
        for job in config['jobs']:
            folder=ROOT/FOLDER/job['id']
            if folder.exists():
                raise FileExistsError('Existing attempt: '+str(folder))
            command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/joint_region_worker_v1.py'),
                '--request',job['request']['path'],'--sha256',job['request']['sha256'],
                '--output',(folder/'result').relative_to(ROOT).as_posix()]
            runtime=execute(ROOT,command,folder,config['policy']['worker_seconds'],config['policy']['worker_memory_MiB'])
            (folder/'process.json').write_bytes(encoded(runtime))
            print(job['id'],runtime['failure'] or 'ok',round(runtime['walltime_seconds'],2),flush=True)
    report=evaluate(ROOT,config)
    for r in report['runs']:print(r['job'],r['texts'],r['links'],flush=True)

if __name__=='__main__':main()
