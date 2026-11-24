"""One leased correction at a time; children never publish database versions."""
import subprocess
import sys
from types import SimpleNamespace
from .dataset_layout_v1 import encoded
from .service_profiles_v2.queue import Queue
from .service_profiles_v2.storage import Storage
from .service_profiles_v2.budget import AttemptBudget
from .web_correction_store_v1 import CorrectionStore
from .web_correction_contract_v1 import decode


def process_once(settings,*,store=None):
    queue=Queue(settings);files=Storage(settings);store=store or CorrectionStore(queue)
    row=store.claim()
    if row is None:return False
    folder=files.path(f"corrections-v1/{row['id']}/{row['owner']}");folder.mkdir(parents=True,exist_ok=False)
    process=budget=None
    try:
        document,versions=queue.job(row['job_id']);parent=queue.run(row['job_id'],row['parent_version'])
        if parent['manifest']['sha256']!=row['source_manifest_sha256'] or versions[-1]['version']!=row['parent_version']:
            raise ValueError('parent_version_changed')
        run={k:parent[k] for k in ('id','job_id','version','status','options','manifest','summary')}
        run={**run,'id':str(run['id']),'job_id':str(run['job_id']),'input_sha256':document['input_sha256']}
        files.manifest(run)
        request={'storage':str(files.root),'folder':str(folder),'stage':row['stage'],'parent':run,
            'input_path':document['input_path'],'page':row['page_index'],'edit':row['request'],
            'approval':row['approval'],'actor':row['actor'],'proposal_id':str(row['id'])}
        path=folder/'request.json';path.write_bytes(encoded(request))
        with (folder/'child.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,'scripts/build_web_correction_v1.py','--request',str(path)],
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            budget=AttemptBudget(process,folder,files,SimpleNamespace(timeout_seconds=60,
                max_output_bytes=128*1024**2,max_output_files=5000),memory_MiB=1024)
            try:code=process.wait(timeout=65)
            except subprocess.TimeoutExpired:raise ValueError('correction_timeout')
            budget.check()
            if code:raise ValueError('correction_validation_or_export_failed')
        summary=decode((folder/'completion.json').read_bytes())
        (folder/'output/resource-budget.json').write_bytes(encoded(budget.measurements()))
        files.manifest(run)
        manifest=files.seal(folder,summary)
        if row['stage']=='preview':store.finish(row,manifest=manifest,summary=summary)
        else:store.publish(row,parent,manifest,summary)
    except Exception as error:
        store.finish(row,error=str(error) if isinstance(error,(ValueError,RuntimeError)) else 'correction_operation_failed')
    finally:
        if budget:budget.stop()
        elif process and process.poll() is None:process.kill();process.wait()
    return True
