"""Common CLI/worker dispatcher with explicit native original-coordinate output."""
from copy import deepcopy
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from .dataset_layout_v1 import digest,encoded,read
from .family_catalog_v2 import require
from .profile_contract_v2 import ProcessingOptions,NATIVE,resolve_profile,validate_binding,verify_models,ProfileError
from .native_input_v2 import input_pages
from .native_profile_kernel_v2 import run as native_run
from .native_merge_v1 import points,vectors

def original_coordinates(result,chain):
    r=deepcopy(result);matrix=np.array(chain['analysis_to_original'])
    for text in r['texts']:text['polygon_px']=points(text['polygon_px'],matrix)
    for g in r['geometry']:
        if g['kind']=='line':g['points_px']=points(g['points_px'],matrix)
        else:
            g['center_px']=points([g['center_px']],matrix)[0]
            if g['kind']=='arc':
                a=math.radians(g['start_angle_deg']);v=vectors([[math.cos(a),math.sin(a)]],matrix)[0]
                g['start_angle_deg']=math.degrees(math.atan2(v[1],v[0]))%360
                g['sweep_angle_deg']*=float(round(np.linalg.det(matrix[:2,:2])))
    for a in r['arrows']:
        a['tip_px']=points([a['tip_px']],matrix)[0];a['direction']=vectors([a['direction']],matrix)[0]
    for a in r['associations']:
        a['anchors_px']=points(a['anchors_px'],matrix) if a['anchors_px'] else []
    for w in r['windows']:w['processed_to_original']=(matrix@np.array(w['processed_to_native'])).tolist()
    r.update(coordinate_system='original encoded raster pixel centres, or original PDF-render pixel centres before explicit rotation',
        coordinate_chain=chain,original_size_px=chain['original_size_px'],
        source_sha256=chain['original_sha256'],analysis_raster_sha256=result['source_sha256'],
        source_size_px=chain['original_size_px'],region=[0,0,*chain['original_size_px']],
        window_boxes_space='analysis raster; processed_to_original maps local observations to original frame')
    return r

def previews(prepared,result,output,prefix):
    prep=read(prepared/'preparation.json');factor=prep['overview']['scale_factor']
    with Image.open(prepared/prep['overview']['path']) as im:base=im.convert('RGB')
    base.save(output/f'{prefix}-source.png')
    geometry=base.copy();d=ImageDraw.Draw(geometry)
    for g in result['geometry']:
        if g['kind']=='line':d.line([(x/factor,y/factor) for x,y in g['points_px']],fill='#00865d',width=1)
        else:
            x,y=g['center_px'];r=g['radius_px'];box=[(x-r)/factor,(y-r)/factor,(x+r)/factor,(y+r)/factor]
            if g['kind']=='circle':d.ellipse(box,outline='#ce651e',width=1)
            else:d.arc(box,start=g['start_angle_deg'],end=g['start_angle_deg']+g['sweep_angle_deg'],fill='#ce651e',width=1)
    geometry.save(output/f'{prefix}-geometry-preview.png')
    text=base.copy();d=ImageDraw.Draw(text)
    for t in result['texts']:
        p=[(x/factor,y/factor) for x,y in t['polygon_px']];d.line([*p,p[0]],fill='#195fd3',width=1)
    text.save(output/f'{prefix}-ocr-preview.png')

def run_profile(source,output,options,*,models=None,expected_binding=None):
    options=ProcessingOptions.model_validate(options);binding=validate_binding(options.profile,expected_binding) if expected_binding is not None else resolve_profile(options.profile)
    if options.profile!=NATIVE:
        from .profile_pipeline_v1 import run_profile as old
        return old(source,output,options.model_dump(),models=models,expected_binding=binding)
    source=Path(source).resolve();output=Path(output).resolve();model_root=verify_models(binding,models)
    if output.exists():raise ProfileError('output_exists','Output exists; choose a new directory')
    sha=digest(source);started=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    from .ocr_engine import LocalOCREngine
    engine=LocalOCREngine(model_root);pages=[]
    for prepared,chain in input_pages(source,output/'private/input',options):
        chain['requested_image_dpi']=options.image_dpi
        index=chain['page_index'];prefix=f'page-{index+1:03d}'
        result=native_run(prepared,output/'private'/prefix,engine)
        original=original_coordinates(result,chain);(output/f'{prefix}-observations.json').write_bytes(encoded(original))
        previews(prepared,result,output,prefix)
        issues=[{'code':'native_observations_unverified','message':'Native-наблюдения требуют проверки; принятые CAD-объекты и DXF этим режимом не создаются.'}]
        if result['deferred_retries']:issues.append({'code':'native_context_budget','message':f"Отложено контекстов у границ: {result['deferred_retries']}."})
        if result['geometry_search_incomplete']:issues.append({'code':'geometry_search_incomplete','message':'Поиск геометрии неполон; пропуск кандидата не доказывает отсутствие объекта.'})
        if result['text_conflicts']:issues.append({'code':'native_text_conflicts','message':f"Конфликтующих пар OCR: {len(result['text_conflicts'])}."})
        if not result['ocr_windows_complete']:issues.append({'code':'ocr_coverage_incomplete','message':'Не все OCR-окна обработаны; сохранены только доступные наблюдения.'})
        pages.append({'page_index':index,'status':'needs_review','source_kind':chain['source_kind'],'issues':issues,
            'coordinate_chain':chain,'native_analysis':{'windows':len(result['windows']),'retry_windows':result['retry_window_count'],
                'deferred_retries':result['deferred_retries'],'accepted_objects':0,'text_count':len(result['texts']),
                'geometry_count':len(result['geometry']),'preview_reduced':True,'preview_coordinate_frame':'analysis raster after EXIF/rotation'},
            'artifacts':{'source_preview':f'{prefix}-source.png','geometry_preview':f'{prefix}-geometry-preview.png',
                         'ocr_preview':f'{prefix}-ocr-preview.png','native_observations':f'{prefix}-observations.json'}})
    if digest(source)!=sha:raise ProfileError('input_integrity','Source changed during native processing')
    require(bool(pages),'No native pages produced')
    report={'status':'needs_review','source_sha256':sha,'page_count':len(pages),'pages':pages,
        'processing_profile':binding,'processing_options':options.model_dump(),'engine':engine.provenance,
        'processing_execution':{'interface_contract':'CLI_and_worker_share_run_profile/2.0','elapsed_seconds':time.perf_counter()-started},
        'limitations':['Native profile exports observations only; use the presentation profile for a separate DXF draft.',
            'Previews are reduced navigation images; processing uses native samples.',
            'Physical units, accuracy, view membership and dimensional constraints remain unverified.']}
    (output/'profile-contract.json').write_bytes(encoded(binding));(output/'report.json').write_bytes(encoded(report))
    return report
