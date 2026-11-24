"""Shared bounded-input binding, retaining each immutable semantic-core binding."""
import hashlib
from . import profile_contract_v5 as previous
from .dataset_layout_v1 import read,checked,encoded
from .raster_stream_v2 import POLICY

ROOT=previous.ROOT
REGISTRY=ROOT/'configs/raster-input-v2-runtime.json'
ProcessingOptions,ProfileError=previous.ProcessingOptions,previous.ProfileError
DEFINITIONS,NATIVE,REVIEW,CANDIDATE=previous.DEFINITIONS,previous.NATIVE,previous.REVIEW,previous.CANDIDATE
verify_models,verify_cv_runtime=previous.verify_models,previous.verify_cv_runtime


def resolve_profile(name):
    core=previous.resolve_profile(name);registry=read(REGISTRY)
    if registry['policy']!=POLICY:raise ProfileError('profile_contract_changed','Raster-input policy changed')
    for ref in registry['code']+[registry['qualification']]:checked(ROOT,ref)
    if not read(checked(ROOT,registry['qualification']))['decoder_qualified']:raise ProfileError('profile_contract_changed','Raster decoder not qualified')
    body={**core,'schema_version':'processing-profile/6.0','core_binding':core,
        'definition':{**core['definition'],'input_adapter':'bounded-raster/2.0'},
        'code':{**core['code'],**{r['path']:r['sha256'] for r in registry['code']}},
        'raster_input_policy':POLICY,'raster_qualification':registry['qualification'],
        'full_resolution_quality_accepted':False}
    body.pop('sha256',None);return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name,binding):
    if binding and binding.get('schema_version')!='processing-profile/6.0':return previous.validate_binding(name,binding)
    value=resolve_profile(name)
    if value!=binding:raise ProfileError('profile_contract_changed','Queued input/core binding changed')
    return value
