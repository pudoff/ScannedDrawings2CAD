"""Bounded, reviewed large-layout annotations for an isolated detector experiment.

This module does not run models. Pixel-role masks, CAD geometry and physical
units remain separate tasks. Validation is diagnostic, never an unseen test.
"""
from collections import Counter, defaultdict
from pathlib import Path
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .architecture_data_readiness_v1 import Boundary
from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from .page_region_review_v1 import READINESS, authenticated_source, native, translations

CLASSES = ['view', 'detail', 'title_block', 'table', 'notes']
READING = 'configs/reviews/layout-region-reading-v1.json'
READING_PIN = {'path':READING,'sha256':'e50e708b3e8713a285dfd9820aa59ffe1fbfd3e238cdd688837f6bd7b78be161'}
OUTPUT = 'artifacts/layout-region-labels-v1'
PREVIOUS = {'path':'artifacts/ocr-model-comparison-result-release-v1.json',
            'sha256':'b36d4c8d401f9741cb1256a0d75a03747afa2d452f3fe50879a184efe3f68913'}
FLAGS = {'human_verified':False,'closed_test_used':False,'independent_evaluation':False,
         'physical_geometry_approved':False,'production_admitted':False}


def box_valid(box, size):
    require(isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box), 'Integer pixel-edge box required')
    x,y,r,b=box
    require(0<=x<r<=size[0] and 0<=y<b<=size[1], 'Layout box outside page')


def native_box(box, factor, size):
    """Area-mean pixel EDGES scale by f, not by the pixel-centre affine."""
    require(type(factor) is int and 1<=factor<=32, 'Invalid integer overview factor')
    result=[min(v*factor,size[i%2]) for i,v in enumerate(box)]
    box_valid(result,size)
    return result


def yolo_line(role, box, factor, overview_size):
    x,y,r,b=[v/factor for v in box]
    w,h=overview_size
    return f'{CLASSES.index(role)} {(x+r)/(2*w):.10f} {(y+b)/(2*h):.10f} {(r-x)/w:.10f} {(b-y)/h:.10f}\n'


def boundary(root):
    readiness=read(checked(root,READINESS))
    groups,sources,one,two,fresh=[read(checked(root,p)) for p in readiness['metadata']]
    return Boundary(groups['groups'],sources,[one,two],
        [{'source_path':s['input'],'source_sha256':s['input_sha256']} for s in fresh['selected_sources']])


def screen_pages(pages, checker):
    """No image/source opening here: finish ALL ancestry checks first."""
    groups=defaultdict(set); hashes=defaultdict(set); ids=set(); result=[]
    for page in pages:
        require(page['id'] not in ids,'Duplicate page id'); ids.add(page['id'])
        row=checker.check(page['source'],page['split'],page['id'])
        require(not row['blocked'] and row['group_id']==page['group_id'],'Blocked/reassigned layout source')
        groups[page['group_id']].add(page['split']); hashes[page['source']['sha256']].add(page['split'])
        result.append(row)
    require(all(len(s)==1 for s in [*groups.values(),*hashes.values()]),'Cross-split layout leakage')
    return result


def validate_reading(value):
    require(value['schema_version']=='layout-region-reading/1.0' and value['classes']==CLASSES,'Layout taxonomy mismatch')
    require(all(value[k] is False for k in ('physical_geometry_approved','independent_evaluation','production_admitted')),'False layout authority')
    require(0<len(value['pages'])<=16,'Layout page budget')
    ids=set()
    for page in value['pages']:
        require(page['split'] in ('train','validation'),'Closed test cannot be annotated here')
        factor=page['factor']; size=page['size_px']; ow,oh=page['overview_size_px']
        require(type(factor) is int and 1<=factor<=32,'Invalid integer overview factor')
        require([math.ceil(v/factor) for v in size]==[ow,oh],'Wrong padded overview grid')
        box_valid(page['frame_hint'],[ow,oh])
        review=page['review']
        require(review['whole_page_layout_reviewed'] is True and review['unknown_target_regions']==[]
                and review['complete_classes']==CLASSES and review['human_verified'] is False
                and review['reviewer']=='assistant_pixel_review','Incomplete/ambiguous page cannot become detector background')
        require(review['scope']=='large_layout_objects_at_authenticated_overview_resolution','Unsupported completeness scope')
        for row in page['regions']:
            require(row['id'] not in ids,'Duplicate layout target'); ids.add(row['id'])
            require(row['role'] in CLASSES and row['observation'],'Missing semantic target decision')
            box_valid(row['overview_box'],[ow,oh]); native_box(row['overview_box'],factor,size)
        # A title stamp's internal grid must not also be a standalone table.
        titles=[r['overview_box'] for r in page['regions'] if r['role']=='title_block']
        for row in page['regions']:
            if row['role']!='table': continue
            a=row['overview_box']
            for b in titles:
                overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                require(overlap/((a[2]-a[0])*(a[3]-a[1]))<0.5,'Stamp grid duplicated as table')
        linked=set()
        for link in page['detail_links']:
            require(all(type(link[k]) is int and 1<=link[k]<=len(page['regions']) for k in ('detail','parent')),'Invalid relation index')
            d=page['regions'][link['detail']-1]; p=page['regions'][link['parent']-1]
            require(d['role']=='detail' and p['role']=='view' and link['label'] and link['printed_ratio'],'Invalid detail relation')
            require(d['id'] not in linked,'Duplicate detail relation'); linked.add(d['id'])
            for field in ('caption','marker'): box_valid(link[field],[ow,oh])
            for small,large in [(link['caption'],d['overview_box']),(link['marker'],p['overview_box'])]:
                require(large[0]<=small[0]<small[2]<=large[2] and large[1]<=small[1]<small[3]<=large[3],
                        'Detail context escapes its labelled region')
        require(linked=={r['id'] for r in page['regions'] if r['role']=='detail'},'Unlinked detail or section misclassified as detail')
        for kind,box in page['non_targets']:
            box_valid(box,[ow,oh])
            require(kind in ('cad_cursor','secondary_coloured_border','projection_method_symbol','blank_layout_negative'),'Unknown non-target disposition')
            if kind=='blank_layout_negative':
                for row in page['regions']:
                    a=row['overview_box']
                    require(min(box[2],a[2])<=max(box[0],a[0]) or min(box[3],a[3])<=max(box[1],a[1]),'Blank negative overlaps a layout target')
    require(len(ids)<=80,'Region annotation budget')


def selection(root, reading_pin):
    require(reading_pin==READING_PIN,'Unsupported layout review version')
    value=read(checked(root,reading_pin)); validate_reading(value)
    screens=screen_pages(value['pages'],boundary(root))
    # No source/overview pixels before the last source has passed screening.
    for page in value['pages']:
        _,prep=authenticated_source(root,page)
        o=prep['overview']; expected={'path':(Path(page['preparation']['path']).parent/o['path']).as_posix(),'sha256':o['sha256']}
        require(page['overview']['sha256']==expected['sha256'] and page['factor']==o['scale_factor']
                and page['overview_size_px']==o['size_px'],'Overview rebound or wrong transform')
        checked(root,page['overview'])
    return value,screens


def native_contacts(root):
    """Display aid only; original native PNG witnesses remain the evidence."""
    folder=root/OUTPUT
    require(not (folder/'manifest.json').exists(),'Cannot rewrite frozen review contacts')
    evidence=read(folder/'evidence.json')
    rows=[w for w in evidence['witnesses'] if '-frame' not in w['id']]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17); contacts=[]
    for offset in range(0,len(rows),12):
        canvas=Image.new('RGB',(2120,1380),'#ddd'); draw=ImageDraw.Draw(canvas)
        entries=[]
        for j,row in enumerate(rows[offset:offset+12]):
            x=(j%4)*530; y=(j//4)*460
            with Image.open(checked(root,row['image'])) as image:
                im=image.convert('RGB'); original=list(im.size); im.thumbnail((520,410))
                draw.text((x+3,y+3),row['id'],font=font,fill='black')
                draw.text((x+3,y+24),f"native {original}, displayed {list(im.size)}",font=font,fill='black')
                canvas.paste(im,(x+5,y+48))
                entries.append({'id':row['id'],'original_size_px':original,'display_size_px':list(im.size)})
        path=folder/f'native-contact-{offset//12+1}.png'; canvas.save(path)
        contacts.append({'image':pin(root,path.relative_to(root).as_posix()),'witnesses':entries})
    (folder/'native-contacts.json').write_bytes(encoded(contacts))
    return contacts


def frame(root,page):
    """Refine manually selected primary frame using black native edge strips.

    Not a frame detector: the reviewer selects which rectangle is the drawing
    frame, excluding outer paper edges and auxiliary inventory strips.
    """
    w,h=page['size_px']; hint=[v*page['factor'] for v in page['frame_hint']]
    left,top,right,bottom=hint; radius=32; edges=[]
    for orientation,coordinate,start,end in [('v',left,top+100,bottom-100),('h',top,left+100,right-100),
                                            ('v',right,top+100,bottom-100),('h',bottom,left+100,right-100)]:
        box=([max(0,coordinate-radius),max(0,start),min(w,coordinate+radius+1),min(h,end)] if orientation=='v' else
             [max(0,start),max(0,coordinate-radius),min(w,end),min(h,coordinate+radius+1)])
        pixels,proof=native(root,page,box,maximum_pixels=2_097_152)
        dark=np.all(pixels<90,axis=2); scores=dark.mean(axis=0 if orientation=='v' else 1)
        peak=int(scores.argmax()); require(float(scores[peak])>0.85,'Frame edge needs manual correction')
        lo=hi=peak
        while lo>0 and scores[lo-1]>=scores[peak]*0.95: lo-=1
        while hi+1<len(scores) and scores[hi+1]>=scores[peak]*0.95: hi+=1
        origin=box[0 if orientation=='v' else 1]; center=origin+(lo+hi)/2
        edges.append({'orientation':orientation,'coordinate':center,'stroke_pixels':[origin+lo,origin+hi+1],
                      'black_coverage':float(scores[peak]),'box':box,'native_proof':proof})
    l,t,r,b=[e['coordinate'] for e in edges]
    require(l<r and t<b,'Inverted native frame')
    return {'source_id':page['id'],'type':'closed_centerline_polyline','vertices':[[l,t],[r,t],[r,b],[l,b],[l,t]],
            'coordinate_system':'native_pixel_centres','filled':False,'dense_mask':False,'detector_class':False,
            'physical_geometry_approved':False,'edges':edges}


def witness(root,page,box,folder,name):
    pixels,proof=native(root,page,box,maximum_pixels=2_097_152)
    path=folder/'witnesses'/f'{name}.png'; Image.fromarray(pixels).save(path)
    return {'id':name,'source_id':page['id'],'box':box,'transform':translations(box),
            'image':pin(root,path.relative_to(root).as_posix()),'native_proof':proof}


def records(value):
    result=[]
    for page in value['pages']:
        for row in page['regions']:
            box=native_box(row['overview_box'],page['factor'],page['size_px'])
            result.append({'id':row['id'],'source_id':page['id'],'role':row['role'],'bbox':box,
                'split':page['split'],'group_id':page['group_id'],'source':page['source'],
                'transform':translations(box),'observation':row['observation'],
                'semantics':'context_envelope_with_dimensions_leaders_and_possible_neighbouring_ink',
                'dense_mask':False,'complete_ocr_context':False,**FLAGS})
    return result


def prepare(root):
    folder=root/OUTPUT; require(not folder.exists(),'Choose a new layout release directory')
    reading_pin=pin(root,READING); value,screens=selection(root,reading_pin)
    folder.mkdir(); (folder/'witnesses').mkdir(); (folder/'overlays').mkdir()
    evidence=[]; outlines=[]; overlays=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    palette=['#d51616','#9c139c','#007a13','#0056db','#a86200']
    for page in value['pages']:
        outline=frame(root,page); outlines.append(outline)
        with Image.open(checked(root,page['overview'])) as im:
            canvas=im.convert('RGB')
        draw=ImageDraw.Draw(canvas)
        for index,row in enumerate(page['regions'],1):
            box=native_box(row['overview_box'],page['factor'],page['size_px']); x,y,r,b=box
            # Native centre context proves source fidelity, not manual review of every pixel.
            cx=(x+r)//2; cy=(y+b)//2
            wb=[max(0,cx-256),max(0,cy-192),min(page['size_px'][0],cx+256),min(page['size_px'][1],cy+192)]
            evidence.append(witness(root,page,wb,folder,row['id']))
            colour=palette[CLASSES.index(row['role'])]
            draw.rectangle(row['overview_box'],outline=colour,width=3)
            draw.text(tuple(row['overview_box'][:2]),f"{index}:{row['role']}",fill=colour,font=font,stroke_width=1,stroke_fill='white')
        for i,link in enumerate(page['detail_links'],1):
            for key in ('caption','marker'):
                evidence.append(witness(root,page,native_box(link[key],page['factor'],page['size_px']),folder,f"{page['id']}-link{i}-{key}"))
        # Four native corner crops establish the selected rectangle's topology.
        for i,(x,y) in enumerate(outline['vertices'][:4],1):
            x=int(x); y=int(y)
            evidence.append(witness(root,page,[max(0,x-100),max(0,y-100),min(page['size_px'][0],x+100),min(page['size_px'][1],y+100)],folder,f"{page['id']}-frame{i}"))
        for i,(kind,box) in enumerate(page['non_targets'],1):
            # Large secondary-border envelope is a disposition, not a negative ROI.
            if kind=='secondary_coloured_border': continue
            evidence.append(witness(root,page,native_box(box,page['factor'],page['size_px']),folder,f"{page['id']}-nontarget{i}"))
        draw.line([(x/page['factor'],y/page['factor']) for x,y in outline['vertices']],fill='#00aaaa',width=2)
        path=folder/'overlays'/f"{page['id']}.png"; canvas.save(path)
        overlays.append(pin(root,path.relative_to(root).as_posix()))
    result={'schema_version':'layout-native-evidence/1.0','reading':reading_pin,'screening':screens,
            'records':records(value),'frames':outlines,'witnesses':evidence,'overlays':overlays,
            'model_calls':0,'review_scope':'whole overviews + native context/caption/frame witnesses; not every native page pixel',**FLAGS}
    (folder/'evidence.json').write_bytes(encoded(result))
    # Native frame witnesses in contact sheets: each crop is shown at 1:1.
    for offset in range(0,len(value['pages']),5):
        batch=value['pages'][offset:offset+5]; canvas=Image.new('RGB',(840,5*235),'#ddd'); draw=ImageDraw.Draw(canvas)
        for j,page in enumerate(batch):
            draw.text((5,j*235),page['id'],font=font,fill='black')
            for i in range(4):
                row=next(w for w in evidence if w['id']==f"{page['id']}-frame{i+1}")
                with Image.open(checked(root,row['image'])) as im: canvas.paste(im,(i*210,j*235+30))
        canvas.save(folder/f'frame-contact-{offset//5+1}.png')
    return {'pages':len(value['pages']),'regions':len(result['records']),'native_witnesses':len(evidence),'frames':len(outlines)}


def release_payload(value,evidence):
    rows=records(value)
    require(evidence['records']==rows,'Evidence/reading target mismatch')
    require([f['source_id'] for f in evidence['frames']]==[p['id'] for p in value['pages']],'Missing or repeated page frame')
    for f in evidence['frames']:
        require(f['type']=='closed_centerline_polyline' and f['vertices'][0]==f['vertices'][-1]
                and len(f['vertices'])==5 and all(f[k] is False for k in ('filled','dense_mask','detector_class','physical_geometry_approved')),
                'Frame outline became an area/physical/detector target')
    links=[]
    for p in value['pages']:
        for l in p['detail_links']:
            links.append({'source_id':p['id'],'detail_id':p['regions'][l['detail']-1]['id'],
                'parent_view_id':p['regions'][l['parent']-1]['id'],'label':l['label'],'printed_ratio':l['printed_ratio'],
                'caption_box':native_box(l['caption'],p['factor'],p['size_px']),
                'marker_box':native_box(l['marker'],p['factor'],p['size_px']),
                'ratio_is_physical_confirmation':False})
    summary={'pages':len(value['pages']),'regions':len(rows),'frames':len(evidence['frames']),
        'detail_links':len(links),'by_role':dict(Counter(r['role'] for r in rows)),
        'by_split':{s:{'pages':sum(p['split']==s for p in value['pages']),
            'groups':len({p['group_id'] for p in value['pages'] if p['split']==s}),
            'regions':sum(r['split']==s for r in rows),
            'by_role':dict(Counter(r['role'] for r in rows if r['split']==s))} for s in ('train','validation')}}
    admission={'schema_version':'layout-detector-admission/1.0','classes':CLASSES,
        'scope':'large_layout_objects_at_authenticated_overview_resolution',
        'train':'isolated_large_layout_detector_experiment_only',
        'validation':'diagnostic_fixed_protocol_only_no_training_or_hyperparameter_selection',
        'page_background':'reviewed_non_target_for_these_five_layout_classes_only',
        'unreviewed_sources':'unknown_not_negative','frame_detector_admitted':False,
        'dense_pixel_training_admitted':False,'full_native_pixel_review_claimed':False,
        'human_review_required_for_production':True,**FLAGS}
    return {'records.json':rows,'frames.json':evidence['frames'],'detail-links.json':links,
            'summary.json':summary,'admission.json':admission,'pages.json':value['pages']}


def validate_acceptance(accepted,evidence,value):
    require(accepted['reviewer']=='assistant_pixel_review' and accepted['human_verified'] is False
            and accepted['unknown_target_regions']==[] and accepted['accepted_page_ids']==[p['id'] for p in value['pages']]
            and accepted['observed_native_witness_ids']==[w['id'] for w in evidence['witnesses']]
            and accepted['observed_overlays']==evidence['overlays'],'Incomplete visual acceptance')


def finalize(root,acceptance_pin):
    folder=root/OUTPUT; require(not (folder/'manifest.json').exists(),'Layout release is immutable')
    accepted=read(checked(root,acceptance_pin)); evidence_pin=accepted['evidence']
    evidence=read(checked(root,evidence_pin)); value,_=selection(root,evidence['reading'])
    validate_acceptance(accepted,evidence,value)
    data=release_payload(value,evidence)
    for name,content in data.items(): (folder/name).write_bytes(encoded(content))
    for split in ('train','validation'):
        (folder/'yolo/images'/split).mkdir(parents=True)
        (folder/'yolo/labels'/split).mkdir(parents=True)
    for page in value['pages']:
        split=page['split']; identifier=page['id']
        shutil.copyfile(checked(root,page['overview']),folder/f'yolo/images/{split}/{identifier}.png')
        lines=''.join(yolo_line(r['role'],r['bbox'],page['factor'],page['overview_size_px']) for r in data['records.json'] if r['source_id']==identifier)
        (folder/f'yolo/labels/{split}/{identifier}.txt').write_text(lines,encoding='utf-8',newline='\n')
    # Root-independent YAML: dataset root must be supplied as an absolute path by the experiment runner.
    yaml='train: images/train\nval: images/validation\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES))
    (folder/'yolo/data.yaml').write_text(yaml,encoding='utf-8',newline='\n')
    filepins={p.relative_to(folder).as_posix():pin(root,p.relative_to(root).as_posix())
              for p in sorted(folder.rglob('*')) if p.is_file()}
    manifest={'schema_version':'layout-region-labels/1.0','reading':evidence['reading'],'acceptance':acceptance_pin,
              'readiness':READINESS,'files':filepins,'model_calls':0,**FLAGS}
    (folder/'manifest.json').write_bytes(encoded(manifest))
    return data['summary.json']


def replay(root,manifest_pin,*,pixels=True):
    manifest=read(checked(root,manifest_pin))
    require(manifest['schema_version']=='layout-region-labels/1.0' and all(manifest[k] is False for k in FLAGS),'Changed release authority')
    value,_=selection(root,manifest['reading'])
    for p in manifest['files'].values(): checked(root,p)
    evidence=read(checked(root,manifest['files']['evidence.json']))
    expected=release_payload(value,evidence)
    for name,content in expected.items():
        require(read(checked(root,manifest['files'][name]))==content,'Repinned derived layout data differ: '+name)
    for page in value['pages']:
        prefix=f"yolo/labels/{page['split']}/{page['id']}.txt"
        text=''.join(yolo_line(r['role'],r['bbox'],page['factor'],page['overview_size_px']) for r in expected['records.json'] if r['source_id']==page['id'])
        require(checked(root,manifest['files'][prefix]).read_text('utf-8')==text,'YOLO coordinates/order differ')
        imagepin=manifest['files'][f"yolo/images/{page['split']}/{page['id']}.png"]
        require(imagepin['sha256']==page['overview']['sha256'],'Export changed authenticated RGB pixels')
    accepted=read(checked(root,manifest['acceptance']))
    require(accepted['evidence']==manifest['files']['evidence.json'],'Rebound evidence')
    validate_acceptance(accepted,evidence,value)
    count=0
    if pixels:
        by_id={p['id']:p for p in value['pages']}
        for row in evidence['witnesses']:
            original,proof=native(root,by_id[row['source_id']],row['box'],maximum_pixels=2_097_152)
            with Image.open(checked(root,row['image'])) as image:
                require(image.mode=='RGB' and np.array_equal(np.asarray(image),original),'Changed native witness pixels')
            require(proof==row['native_proof'],'Changed native witness provenance'); count+=1
        require([frame(root,p) for p in value['pages']]==evidence['frames'],'Native frame reconstruction differs')
    return {**expected['summary.json'],'native_witnesses_replayed':count,'pixels_replayed':pixels,
            'model_calls':0,**FLAGS}


def load(root,manifest_pin,*,split,purpose):
    """Explicit scoped admission before even reading the release metadata."""
    allowed={'train':'large_layout_detector_experiment',
             'validation':'diagnostic_large_layout_detector_evaluation'}
    require(split in allowed and purpose==allowed[split],'Forbidden layout release use')
    replay(root,manifest_pin,pixels=False)
    manifest=read(checked(root,manifest_pin))
    rows=read(checked(root,manifest['files']['records.json']))
    pages=read(checked(root,manifest['files']['pages.json']))
    return {'split':split,'purpose':purpose,'records':[r for r in rows if r['split']==split],
            'pages':[p for p in pages if p['split']==split],
            'yolo_root':(root/OUTPUT/'yolo').resolve().as_posix(),
            'admission':read(checked(root,manifest['files']['admission.json']))}
