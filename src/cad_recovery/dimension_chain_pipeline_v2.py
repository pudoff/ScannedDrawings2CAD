"""c.11 pipeline with explicit producer-versus-publication status metadata."""
from .dimension_chain_pipeline_v1 import run as original_run,ARMS
from .completeness_publication_v1 import clarify

def run(*args,**kwargs):
    result=original_run(*args,**kwargs)
    result['algorithm_version']='dimension-chain-pipeline/2.0'
    for arm in ('gate_only','consistency_and_gate'):result['arms'][arm]=clarify(result['arms'][arm])
    return result
