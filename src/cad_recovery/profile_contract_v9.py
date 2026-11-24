"""Version passive search coverage separately from the identical v8 core."""
import hashlib
from . import profile_contract_v8 as previous
from .dataset_layout_v1 import checked,digest,encoded,read
ROOT=previous.ROOT
REGISTRY=ROOT/'configs/search-completeness-web-v1.json'
ProcessingOptions,ProfileError=previous.ProcessingOptions,previous.ProfileError
DEFINITIONS,NATIVE,REVIEW,CANDIDATE=previous.DEFINITIONS,previous.NATIVE,previous.REVIEW,previous.CANDIDATE
verify_models,verify_cv_runtime=previous.verify_models,previous.verify_cv_runtime


def resolve_profile(name):
    core=previous.resolve_profile(name);registry=read(REGISTRY)
    from .search_completeness_v1 import POLICY
    if registry.get('schema_version')!='search-reporting-web/1.0' or registry.get('policy')!=POLICY:
        raise ProfileError('profile_contract_changed','Search reporting registry differs')
    for p in registry['code']:checked(ROOT,p)
    body={**core,'schema_version':'processing-profile/9.0','core_binding':core,
        'search_reporting':dict(version='search-completeness/1.0',quality_credit=False),
        'code':{**core['code'],**{r['path']:r['sha256'] for r in registry['code']}},'search_reporting_registry_sha256':digest(REGISTRY)}
    body.pop('sha256',None);return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}


def validate_binding(name,value):
    if value and value.get('schema_version')!='processing-profile/9.0':return previous.validate_binding(name,value)
    expected=resolve_profile(name)
    if value!=expected:raise ProfileError('profile_contract_changed','Queued search reporting binding changed')
    return expected
