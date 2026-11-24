"""Separate bounded child process per leased run; stale attempts cannot publish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .storage import Storage,write_json,digest
from .queue import Queue
from .config import ProcessingOptions
from .process import ChildGuard

class LeaseLost(RuntimeError):pass

def process_once(settings,queue=None,storage=None,*,runner=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    if hasattr(repo,'bind_storage'):repo.bind_storage()
    run=repo.claim()
    if run is None:return False
    process=None;guard=None
    try:
        source=files.path(run['input_path'])
        if digest(source)!=run['input_sha256']:raise ValueError('input_integrity')
        options=ProcessingOptions.model_validate(run['options']);folder=files.attempt(run)
        request={'source':str(source),'input_sha256':run['input_sha256'],'output':str(folder/'output'),
            'options':options.model_dump()};write_json(folder/'request.json',request)
        script=Path(runner) if runner else Path(__file__).resolve().parents[3]/'scripts/service_pipeline_v1.py'
        env=dict(os.environ);env['XDG_CACHE_HOME']=str(folder/'.cache')
        tick=time.monotonic();heartbeat=tick+settings.lease_seconds/3
        with (folder/'process.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(script),'--request',str(folder/'request.json')],
                stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            guard=ChildGuard(process)
            while process.poll() is None:
                now=time.monotonic()
                if now-tick>settings.timeout_seconds:raise TimeoutError('processing_timeout')
                total,count=files.output_size(folder)
                if total>settings.max_output_bytes or count>settings.max_output_files:raise OverflowError('output_limit')
                if now>=heartbeat:
                    if not repo.heartbeat(run):raise LeaseLost('lease_lost')
                    heartbeat=now+settings.lease_seconds/3
                time.sleep(.1)
        if process.returncode!=0:
            result=json.loads((folder/'status.json').read_text('utf-8')) if (folder/'status.json').exists() else {}
            raise RuntimeError(result.get('code','processing_failed'))
        report=json.loads((folder/'output/report.json').read_text('utf-8'))
        summary={'status':report['status'],'page_count':report['page_count'],
            'pages':[{'page_index':p['page_index'],'status':p['status'],'issues':p['issues']} for p in report['pages']],
            'limitations':report['limitations']}
        manifest=files.seal(folder,summary)
        # finish() checks token AND unexpired database lease in the same UPDATE.
        repo.finish(run,manifest=manifest,summary=summary)
    except LeaseLost:
        pass # Lost leases never change another owner's row or publish their output.
    except Exception as error:
        code='output_limit' if isinstance(error,OverflowError) else str(error) if str(error) in ('input_integrity','processing_timeout','output_limit','input_invalid','processing_failed') else 'processing_failed'
        repo.finish(run,error={'code':code,'message':'Processing stopped; inspect local worker attempt logs'})
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10)
        if guard is not None:guard.close()
    return True
