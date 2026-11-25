"""One new-submission policy; stored document bindings remain read-only."""
import inspect,json,re
from uuid import UUID
from fastapi import HTTPException,Query,Request
from fastapi.responses import HTMLResponse,Response
from fastapi.routing import APIRoute

DEFAULT='native_cad_layers_draft'
EXCLUDED='native_cad_review_demo'
SCHEMA='clean-demo-policy/1.0'
UNAVAILABLE='Профиль с подготовленным примером размера недоступен для новых запусков. Выберите профиль CAD · целые объекты и рамка.'


def allowed(profile):
    if profile==EXCLUDED:
        raise HTTPException(409,dict(code='profile_unavailable',message=UNAVAILABLE))
    return profile


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Clean demo UI hook differs: '+old[:90])
    return source.replace(old,new,1)


def index_html(source):
    source,count=re.subn(r'<option\b[^>]*\bvalue="'+EXCLUDED+r'"[^>]*>.*?</option>','',source,flags=re.S)
    if count!=1:raise ValueError('Prepared profile selector hook differs')
    source=re.sub(r'(<option\b[^>]*)\sselected(?=[\s>])',r'\1',source)
    source=once(source,'<option value="'+DEFAULT+'">','<option value="'+DEFAULT+'" selected>')
    return once(source,'<select id="profile">',
        '<p id="profile-policy-note" class="field-hint" role="status" hidden></p><select id="profile" aria-describedby="profile-policy-note">')


def javascript(source):
    helpers='''  const cleanDefault = 'native_cad_layers_draft', unavailableProfile = 'native_cad_review_demo';
  function setProcessingProfile(name) {
    const chooser=$('profile');
    for(const option of [...chooser.options])if(option.dataset.archived)option.remove();
    const archived=name===unavailableProfile || ![...chooser.options].some(o=>o.value===name);
    if(archived){const option=new Option('Архивный профиль · перерасчёт недоступен',name);option.disabled=true;option.dataset.archived='true';chooser.add(option);}
    chooser.value=name;
    const note=$('profile-policy-note');note.hidden=!archived;
    note.textContent=archived?'Показан сохранённый результат. Для новой обработки выберите доступный профиль.':'';
  }
  $('profile').addEventListener('change',()=>setProcessingProfile($('profile').value));
'''
    source=once(source,'  function options() {',helpers+'''  function options() {
    if($('profile').value===unavailableProfile || $('profile').selectedOptions[0]?.disabled)
      throw new Error('Этот архивный профиль доступен только для просмотра. Выберите доступный профиль для новой обработки.');''')
    source=once(source,"$('profile').value = o.profile;","setProcessingProfile(o.profile);")
    source=once(source,"    state.file = file;", "    if($('profile').selectedOptions[0]?.disabled)setProcessingProfile(cleanDefault);\n    state.file = file;")
    return source


async def version_request(request):
    """Bound the same 4KiB body; replay to the original version endpoint."""
    body=bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body)>4096:raise HTTPException(413,'Options limit exceeded')
    raw=bytes(body)
    try:data=json.loads(raw or b'{}')
    except (ValueError,UnicodeError):data=None
    if isinstance(data,dict):
        allowed(data.get('profile'))
        if 'profile' not in data:
            data['profile']=DEFAULT;raw=json.dumps(data,separators=(',',':'),ensure_ascii=True).encode()
    emitted=False
    async def receive():
        nonlocal emitted
        if not emitted:
            emitted=True;return dict(type='http.request',body=raw,more_body=False)
        return await request.receive()
    return Request(request.scope,receive=receive)


def install(api):
    from .profile_contract_v9 import DEFINITIONS,resolve_profile
    def route(path,method):
        rows=[r for r in api.routes if getattr(r,'path',None)==path and method in getattr(r,'methods',())]
        if len(rows)!=1:raise ValueError('Clean demo API hook differs: '+path)
        return rows[0]
    upload=route('/v1/jobs','POST');version=route('/v1/jobs/{job_id}/versions','POST');profiles=route('/v1/profiles','GET')
    api.router.routes=[r for r in api.routes if r not in (upload,version,profiles)]
    @api.get('/v1/profiles')
    def processing_profiles():
        return dict(schema_version='processing-profile-list/9.0',default=DEFAULT,
            profiles=[resolve_profile(n) for n in DEFINITIONS if n!=EXCLUDED],
            submission_policy=dict(schema_version=SCHEMA,excluded=[EXCLUDED],stored_results_readable=True,
                excluded_reprocessing='explicit compatible profile required'))
    @api.post('/v1/jobs',status_code=202)
    async def new_upload(request:Request,filename:str=Query(min_length=1,max_length=180),
                         profile:str=DEFAULT,render_dpi:float=120.,image_dpi:float|None=None,rotation_deg:int=0):
        allowed(profile)
        return await upload.endpoint(request,filename,profile,render_dpi,image_dpi,rotation_deg)
    @api.post('/v1/jobs/{job_id}/versions',status_code=202)
    async def new_version(request:Request,job_id:UUID):
        return await version.endpoint(await version_request(request),job_id)
    api.openapi_schema=None
    return api


def install_web(web):
    for path,transform in (('/',index_html),('/demo/assets/app.js',javascript)):
        rows=[r for r in web.routes if getattr(r,'path',None)==path and hasattr(r,'endpoint')]
        if len(rows)!=1:raise ValueError('Clean demo Web hook differs: '+path)
        old=rows[0]
        def endpoint(old,transform,path):
            async def asset():
                response=old.endpoint()
                if inspect.isawaitable(response):response=await response
                headers={k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-type','etag')}
                cls=HTMLResponse if path=='/' else Response
                return cls(transform(response.body.decode('utf-8')),media_type='text/html' if path=='/' else 'text/javascript',headers=headers)
            return asset
        web.router.routes=[r for r in web.routes if r is not old]
        web.router.routes.insert(0,APIRoute(path,endpoint(old,transform,path),methods=['GET']))
    return web
