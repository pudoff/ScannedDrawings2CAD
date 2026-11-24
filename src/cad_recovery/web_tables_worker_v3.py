"""Reuse the fenced table queue; dispatch retained bindings and durable diagnostics."""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from . import web_tables_worker_v2 as previous
from .web_tables_store_v1 import TableStore
from .service_profiles_v11.queue import Queue
from .web_tables_contract_v1 import binding as first
from .web_tables_contract_v2 import binding as second
from .web_tables_contract_v3 import binding
from .durable_attempt_budget_v1 import AttemptBudget
from .service_profiles_v12.worker import computation_running
from .dataset_layout_v1 import read,encoded


def process_once(settings,*,store=None,runner=None):
    store=store or TableStore(Queue(settings));state={};budgets=[]
    class Store:
        def __getattr__(self,name):return getattr(store,name)
        def claim(self):
            row=store.claim();state['row']=row;return row
    def selected():
        value=state['row']['request']['binding_sha256']
        for factory in (first,second,binding):
            result=factory()
            if result['sha256']==value:return result
        raise ValueError('Unsupported table extension')
    def serialize(value):
        if isinstance(value,dict) and value.get('edit',{}).get('text_selection'):
            selection=value['edit']['text_selection'];row=state['row']
            source=store.get(row['job_id'],row['parent_version'],row['page_index'],selection['proposal_id'])
            if source['status']!='prepared' or source['preview_manifest']['sha256']!=selection['preview_sha256']:
                raise ValueError('Text source proposal changed')
            value={**value,'text_source_manifest':source['preview_manifest'],'text_source_summary':source['summary']}
        return encoded(value)
    class Budget(AttemptBudget):
        def __init__(self,*a,**k):super().__init__(*a,**k);budgets.append(self)
    script=runner or Path(__file__).resolve().parents[2]/'scripts/build_web_tables_v3.py'
    try:
        with ExitStack() as stack:
            for name,value in [('binding',selected),('AttemptBudget',Budget),('computation_running',computation_running),('encoded',serialize)]:
                stack.enter_context(patch.object(previous,name,value))
            return previous.process_once(settings,store=Store(),runner=script)
    finally:
        for budget in budgets:
            errorfile=budget.folder/'worker-error.json'
            if errorfile.exists():budget.fail(read(errorfile).get('error','table_processing_failed')[:200])
            else:budget.finish('completed')
            budget.stop()
