"""V9 summaries carry frame-bound coverage; historical summaries stay exact."""
from pathlib import Path
from types import SimpleNamespace
from . import sealed_child_v1 as previous
from .runtime_worker_v2 import child_main as correlated_child
from .runtime_io_v1 import read_json
from .service_profiles_v9.storage import Storage
from .search_completeness_v1 import validate
from .dataset_layout_v1 import digest


def add_coverage(summary,report,output):
    pages={p['page_index']:p for p in report['pages']}
    for page in summary['pages']:
        original=pages[page['page_index']];value=original.get('search_completeness')
        if value is None:continue
        path=Path(output)/'private'/f"page-{page['page_index']+1:03d}"/'experimental/gray.png'
        validate(value,report,original,working_image_sha256=digest(path) if path.exists() else None)
        page['search_completeness']=value
    return summary


def finalize(request,progress,*,seal=None):
    if request.get('profile_binding',{}).get('schema_version')!='processing-profile/9.0':
        return previous.finalize(request,progress,seal=seal)
    def enriched(folder,summary):
        report=read_json(Path(folder)/'output/report.json',limit=16*1024*1024)
        add_coverage(summary,report,Path(folder)/'output')
        return (seal or Storage(SimpleNamespace(root=Path(request['_artifact_storage']))).seal)(folder,summary)
    return previous.finalize(request,progress,seal=enriched)


def child_main(path,operation,*,seal=None):
    def execute(request,progress):
        operation(request,progress);finalize(request,progress,seal=seal)
    return correlated_child(path,execute)
