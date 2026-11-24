"""Bind un-applied TEXT previews to an exact parent without importing CAD."""
from copy import deepcopy
import hashlib
from .dataset_layout_v1 import digest,encoded
from .web_text_publication_v1 import state_path


def parent_anchor(parent,document,page):
    return dict(job_id=str(parent['job_id']),run_id=str(parent['id']),version=parent['version'],
        page_index=page,source_sha256=document['input_sha256'],manifest=dict(parent['manifest']))


def validate_proposal(state,parent,document,page,original,current):
    lineage=state.get('preview_lineage')
    if lineage is None:
        # Compatibility is deliberately restricted to an untouched legacy
        # evidence-only proposal. A changed legacy preview must first be applied.
        if digest(original)!=digest(current):
            raise ValueError('Legacy changed TEXT preview requires apply or new evidence')
        return
    if (not isinstance(lineage,dict) or lineage.get('schema_version')!='text-preview-lineage/1.0'
            or lineage.get('parent')!=parent_anchor(parent,document,page)
            or lineage.get('basis_sha256')!=state['basis']['baseline_sha256']):
        raise ValueError('TEXT preview belongs to another exact parent')


def attach_lineage(state,request,*,previous_output=None):
    value=deepcopy(state)
    previous=None
    if previous_output is not None:
        choice=request['edit']['text_selection']
        previous=dict(manifest=dict(request['text_source_manifest']),source_kind=choice['source_kind'],
            proposal_id=str(choice['proposal_id']),state_sha256=digest(state_path(previous_output,request['page'])),
            selection_sha256=hashlib.sha256(encoded(choice)).hexdigest())
    value['preview_lineage']=dict(schema_version='text-preview-lineage/1.0',
        parent=parent_anchor(request['parent'],request['document'],request['page']),
        basis_sha256=state['basis']['baseline_sha256'],previous=previous)
    return value
