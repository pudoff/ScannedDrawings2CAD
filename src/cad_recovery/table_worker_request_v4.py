"""Retain v1-v5 budgets/producers; route v6 to the registered TEXT wrapper."""
from .table_worker_request_v1 import prepare_table_execution as base
from .table_worker_request_v3 import prepare_table_execution as previous
from .web_tables_contract_v7 import binding


def prepare_table_execution(row,parent,document,*,factories,**kwargs):
    if row['request']['binding_sha256']==binding()['sha256']:
        return base(row,parent,document,factories=factories,**kwargs)
    return previous(row,parent,document,factories=factories[:-1],**kwargs)
