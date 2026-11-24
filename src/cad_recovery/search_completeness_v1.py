"""Four-state execution coverage. Successful output never implies full search."""
from copy import deepcopy
from contextlib import ExitStack,contextmanager
from threading import current_thread,main_thread
from unittest.mock import patch
import hashlib
from .dataset_layout_v1 import encoded
from .grid_role_revision_v1 import sha

STAGES=('seed','width','axis','eligibility','localized','context','curved')
POLICY=dict(version='search-completeness/1.0',states=['complete','incomplete','unknown','not_applicable'],
    structural_cv='recorded source path/boundary graph flag, including initial roles; not all CV stages',
    legacy_cv='unchanged experimental aggregate; historically excludes seed',
    head_search='seed AND width/axis/eligibility/localized/context/curved',
    overall='legacy_cv AND structural_cv AND head_search; not detection recall',
    unknown_historical_fields='unknown; never inferred from DXF or other flags',quality_credit=False)


def combine(values):
    states=[r['status'] for r in values]
    if 'incomplete' in states:status='incomplete'
    elif 'unknown' in states or not states:status='unknown'
    elif all(s=='not_applicable' for s in states):status='not_applicable'
    else:status='complete'
    return dict(status=status,reasons=sorted({reason for r in values for reason in r['reasons']}))


def limits(value):
    found=set()
    def visit(item):
        if not isinstance(item,dict):return
        for key,v in item.items():
            if key=='limits' and isinstance(v,list):found.update(x for x in v if isinstance(x,str))
            elif isinstance(v,dict):visit(v)
    visit(value);return sorted(found)


def observed(stage,result,key,invert=False):
    value=result.get(key);valid=type(value) is bool
    status='unknown' if not valid else 'complete' if value!=invert else 'incomplete'
    reasons=[] if status=='complete' else limits(result)
    if not reasons and status!='complete':reasons=['producer_reported_incomplete' if valid else 'missing_explicit_producer_flag']
    return dict(status=status,reasons=[stage+':'+x for x in reasons])


def unknown(reason='not_recorded_in_this_version'):
    return dict(status='unknown',reasons=[reason])


def domains(stages,legacy,structural):
    additions=combine([stages[k] for k in STAGES if k!='seed']);heads=combine([stages['seed'],additions])
    return dict(structural_cv=structural,legacy_cv=legacy,head_seed=stages['seed'],head_additions=additions,
                head_search=heads,overall=combine([legacy,structural,heads]))


def frame(report,page):
    return dict(coordinate_system='experimental_working_raster_pixel_centres',page_index=page['page_index'],
        working_size_px=page.get('experimental_cv',{}).get('size_px'),
        input_to_working=page.get('experimental_cv',{}).get('input_to_working'),
        preprocessing_transforms=page.get('preprocessing',{}).get('transforms'),
        source_pixel_to_analysis_px=report.get('common_input',{}).get('source_pixel_to_analysis_px'))


def fingerprint(value):return hashlib.sha256(encoded(value)).hexdigest()


@contextmanager
def capture():
    """Passive hooks confined to one serial CLI/worker child; no second search."""
    if current_thread() is not main_thread():raise ValueError('Coverage capture requires a serial child')
    from . import stroke_roles_v2 as seed
    from . import local_width_heads_v1 as width
    from . import local_axis_heads_v2 as axis
    from . import head_eligibility_v2 as eligibility
    from . import experimental_cv_v1 as cv
    import run_head_recall_v1 as producer
    records={};active=[]
    def wrap(stage,function,key,invert=False):
        def call(*args,**kwargs):
            result=function(*args,**kwargs)
            if active:active[-1]['calls'].setdefault(stage,[]).append(dict(pixel_sha256=sha(args[0]),state=observed(stage,result,key,invert)))
            return result
        return call
    old_infer=producer.infer
    def infer(gray,*args,**kwargs):
        result=old_infer(gray,*args,**kwargs)
        if active:
            for stage,value,key,inv in [('localized',result[0],'search_incomplete',True),('context',result[1],'complete',False),('curved',result[2],'complete',False)]:
                active[-1]['calls'].setdefault(stage,[]).append(dict(pixel_sha256=sha(gray),state=observed(stage,value,key,inv)))
            active[-1]['guarded_sha256']=fingerprint(result[3]['guarded'])
        return result
    old_analyze=cv.analyze
    def analyze(gray_path,folder):
        row=dict(calls={});active.append(row)
        try:
            result=old_analyze(gray_path,folder);pixels=sha(result['gray']);stages={}
            for stage in STAGES:
                calls=row['calls'].get(stage,[])
                stages[stage]=calls[0]['state'] if len(calls)==1 and calls[0]['pixel_sha256']==pixels else unknown(stage+':missing_or_ambiguous_frame_bound_call')
            row.update(stages=stages,working_pixels_sha256=pixels);records[str(folder)]=row
            return result
        finally:active.pop()
    with ExitStack() as stack:
        for module,name,value in [(seed,'search',wrap('seed',seed.search,'search_incomplete',True)),
                (width,'search',wrap('width',width.search,'search_incomplete',True)),
                (axis,'guarded_search',wrap('axis',axis.guarded_search,'search_incomplete',True)),
                (eligibility,'apply',wrap('eligibility',eligibility.apply,'search_incomplete',True)),
                (producer,'infer',infer),(cv,'analyze',analyze)]:
            stack.enter_context(patch.object(module,name,value))
        yield records


def make(report,page,record,*,gray_sha256=None,graph=None,binding_sha256=None):
    stages={k:unknown() for k in STAGES} if record is None else deepcopy(record['stages'])
    legacy=observed('legacy_cv',page.get('experimental_cv',{}),'search_complete')
    structural=observed('structural_cv',graph or {},'search_incomplete',True)
    result=dict(schema_version=POLICY['version'],policy=POLICY,source_sha256=report['source_sha256'],
        page_index=page['page_index'],profile_sha256=report['processing_profile']['sha256'],
        binding_sha256=binding_sha256,frame=frame(report,page),working_image_sha256=gray_sha256,
        working_pixels_sha256=None if record is None else record['working_pixels_sha256'],
        guarded_sha256=None if record is None else record.get('guarded_sha256'),
        stages=stages,domains=domains(stages,legacy,structural),quality_credit=False)
    if page.get('experimental_cv',{}).get('branch')=='direct_vector_pdf':
        na=dict(status='not_applicable',reasons=['direct_vector_branch_no_raster_head_search'])
        result['stages']={k:deepcopy(na) for k in STAGES};result['domains']=domains(result['stages'],na,na)
    result['frame_sha256']=fingerprint(result['frame']);return result


def validate(value,report,page,*,working_image_sha256=None):
    if value['schema_version']!=POLICY['version'] or value['policy']!=POLICY or value['quality_credit'] is not False:raise ValueError('Coverage contract differs')
    if value['source_sha256']!=report['source_sha256'] or value['page_index']!=page['page_index'] or value['profile_sha256']!=report['processing_profile']['sha256']:raise ValueError('Coverage source/profile/page differs')
    expected_binding=report['processing_profile'].get('search_reporting_registry_sha256')
    if value['binding_sha256']!=expected_binding:raise ValueError('Coverage reporting binding differs')
    if value['frame']!=frame(report,page) or value['frame_sha256']!=fingerprint(value['frame']):raise ValueError('Coverage frame differs')
    if working_image_sha256 is not None and value['working_image_sha256']!=working_image_sha256:raise ValueError('Coverage raster differs')
    for r in [*value['stages'].values(),*value['domains'].values()]:
        if r['status'] not in POLICY['states'] or not isinstance(r['reasons'],list) or not all(isinstance(x,str) for x in r['reasons']):raise ValueError('Invalid coverage state')
    expected=domains(value['stages'],value['domains']['legacy_cv'],value['domains']['structural_cv'])
    if value['domains']!=expected:raise ValueError('Coverage aggregate differs')
    return value
