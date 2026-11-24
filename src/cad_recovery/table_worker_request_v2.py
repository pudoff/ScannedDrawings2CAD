"""Retain v1-v4 requests; dispatch exact-parent chain requests as v5."""
from .table_worker_request_v1 import prepare_table_execution as previous
from .web_tables_contract_v5 import binding


def prepare_table_execution(row,parent,document,*,factories,**kwargs):
    # The previous helper classifies the last factory as text-only. Keep v4's
    # original limits when draining old queued requests in the new worker.
    selected=factories if row['request']['binding_sha256']==binding()['sha256'] else factories[:-1]
    return previous(row,parent,document,factories=selected,**kwargs)
