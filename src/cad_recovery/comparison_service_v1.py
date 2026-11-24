"""Read-only, context-bound comparison derivation with one bounded render child."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .comparison_frame_v1 import frame,svg_to_canvas
from .service_profiles_v3.inspection import Inspector,InspectionLimit
from .observation_graph_v1 import ObservationLimit
from .service_profiles_v2.budget import AttemptBudget

ROOT=Path(__file__).resolve().parents[2]


class ComparisonLimit(ValueError):pass


class Comparisons:
    def __init__(self,storage):
        self.files=storage;self.inspector=Inspector(storage);self.lock=Lock()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dxf-comparison')
        self.jobs=OrderedDict();self.children=set();self.stopped=False
        self.work=storage.path('comparison-cache-v1');self.work.mkdir(exist_ok=True)

    def close(self):
        with self.lock:
            self.stopped=True;children=list(self.children)
        for child in children:
            if child.poll() is None:child.terminate()
        self.pool.shutdown(wait=False,cancel_futures=True)

    def _render(self,path,digest,render_box):
        with tempfile.TemporaryDirectory(prefix='render-',dir=self.work) as directory:
            folder=Path(directory)
            if not folder.resolve().is_relative_to(self.work.resolve()):raise ValueError('Invalid derived directory')
            request=folder/'request.json';output=folder/'render.json'
            request.write_text(json.dumps({'path':str(path),'sha256':digest,'render_box':render_box}),encoding='utf-8')
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/render_comparison_v1.py'),
                '--request',str(request),'--output',str(output)],cwd=ROOT,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            budget=None
            try:
                budget=AttemptBudget(child,folder,self.files,SimpleNamespace(timeout_seconds=20,
                    max_output_bytes=32*1024**2,max_output_files=4),memory_MiB=512)
                with self.lock:
                    if self.stopped:child.terminate()
                    self.children.add(child)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait();return {'status':'unavailable','reason':'render_timeout'}
                if code or not output.is_file():return {'status':'unavailable','reason':'unsupported_or_limited_dxf'}
                budget.check()
                if output.stat().st_size>32*1024**2:return {'status':'unavailable','reason':'render_output_limit'}
                value=json.loads(output.read_text(encoding='utf-8'))
                if value['dxf_sha256']!=digest:raise ValueError('Render context mismatch')
                value['render_budget']=budget.measurements()
                value['status']='ready' if 'svg' in value else 'counts_only';return value
            finally:
                if budget is not None:budget.stop()
                if child.poll() is None:child.kill();child.wait()
                with self.lock:self.children.discard(child)

    def describe(self,run,page):
        manifest_path,manifest=self.files.manifest(run)  # revalidate even on cache hits
        count=manifest['summary'].get('page_count',0)
        if type(page)is not int or not 0<=page<count:raise KeyError('Page missing')
        root=manifest_path.parent;entries=manifest['files'];prefix=f'page-{page+1:03d}'
        def entry(name):return next((e for e in entries if e['filename']==name),None)
        def path(e):return self.files.path((root/e['path']).relative_to(self.files.root).as_posix())
        def data(name):return self.inspector._json(root,entries,name)
        report=data('report.json')
        if report.get('source_sha256')!=run.get('input_sha256'):raise ValueError('Comparison input mismatch')
        current=report['pages'][page]
        graph=data(prefix+'-graph.json') if entry(prefix+'-graph.json') else {'page':{'kind':'raster'},'primitives':[]}
        try:inspection=self.inspector.inspect(run,page)
        except (InspectionLimit,ObservationLimit) as exc:raise ComparisonLimit(str(exc)) from exc
        except LookupError as exc:
            if type(exc)is not LookupError:raise
            inspection=None
        source=entry(prefix+'-source.png')
        if source is None:raise LookupError('Source preview missing')
        with Image.open(path(source)) as image:size=list(image.size)
        # Native navigation pixels are already in analysis/preview orientation.
        if inspection and inspection.get('source_coordinate_chain'):
            source=next(e for e in entries if e['id']==inspection['preview']['artifact_id'])
            with Image.open(path(source)) as image:size=list(image.size)
        export=current.get('demo_export',{})
        try:frames=frame(current,graph,size,inspection,export.get('requested_rotation_clockwise_deg',0))
        except LookupError:
            frames={'canvas_size_px':size,'source_to_canvas':np.eye(3).tolist(),
                'observation_to_canvas':np.eye(3).tolist(),'dxf_to_canvas':None,'render_box':None,
                'source_label':'Исходник; совмещение DXF недоступно'}
        context={'job_id':str(run['job_id']),'version':run['version'],'page_index':page,
                 'manifest_sha256':run['manifest']['sha256'],'input_sha256':run['input_sha256'],
                 'profile':run['options']['profile']}
        if inspection is None:
            inspection={'objects':[],'associations':[],'connectivity':{'endpoints':[],'contacts':[],'complete':True},
                'counts':{'objects':0,'associations':0,'accepted':0},'source_conflicts':[],
                'source_completeness':{'observations_available':False},'source_notations':[],
                'source_coordinate_chain':{},'context':context,'physical_scale_confirmed':False,
                'preview':{'artifact_id':source['id'],'size_px':frames['canvas_size_px'],
                    'observation_to_preview':frames['observation_to_canvas']}}
        value={'schema_version':'comparison/1.0','context':context,'inspection':inspection,'frame':frames,
            'source':{**{k:source[k] for k in ['id','sha256','filename']},'size_px':size},
            'dxf':{'status':'absent'},'source_completeness':inspection['source_completeness'],
            'verified_geometry_counts':{'primitives':0,'dimensions':0},'export_report':export,
            'quality_status':'needs_review','acceptance':'No geometry acceptance occurs in the viewer'}
        value['masks']={}
        for role,suffix,ink_value in [('original','-binary.png',0),('cleaned','-cleaned-binary.png',0),('removed','-removed-mask.png',255)]:
            mask=entry(prefix+suffix)
            if mask is None:continue
            with Image.open(path(mask)) as image:mask_size=list(image.size)
            if mask_size==frames['canvas_size_px'] and mask_size[0]*mask_size[1]<=4_000_000:
                value['masks'][role]={k:mask[k] for k in ['id','sha256','filename']}
                value['masks'][role].update(size_px=mask_size,ink_value=ink_value)
        dxf=entry(prefix+'.dxf')
        if dxf is None:return value
        value['dxf'].update({k:dxf[k] for k in ['id','sha256','filename']})
        if export.get('dxf_sha256',dxf['sha256'])!=dxf['sha256']:raise ValueError('Export report/DXF fingerprint mismatch')
        if dxf['size_bytes']>16*1024**2:
            value['dxf'].update(status='unavailable',reason='dxf_byte_limit');return value
        key=(str(run['job_id']),run['version'],page,run['manifest']['sha256'],dxf['sha256'],'comparison/1.0')
        with self.lock:
            if key not in self.jobs:
                if self.stopped or sum(not f.done() for f in self.jobs.values())>=2:
                    value['dxf'].update(status='busy');return value
                self.jobs[key]=self.pool.submit(self._render,path(dxf),dxf['sha256'],frames['render_box'])
            self.jobs.move_to_end(key);future=self.jobs[key]
            # Completed images occupy at most 32MiB; never evict active children.
            def retained_bytes():
                return sum(len(f.result().get('svg','').encode()) for f in self.jobs.values() if f.done() and not f.exception())
            while len(self.jobs)>8 or retained_bytes()>32*1024**2:
                old=next((k for k,f in self.jobs.items() if k!=key and f.done()),None)
                if old is None:break
                del self.jobs[old]
        if not future.done():value['dxf'].update(status='pending');return value
        try:rendered=future.result()
        except Exception:rendered={'status':'unavailable','reason':'render_failed'}
        value['dxf'].update({k:v for k,v in rendered.items() if k!='svg'})
        value['dxf']['geometry_unrecovered']=export.get('empty',not bool(graph.get('primitives')))
        if rendered['status']=='counts_only':value['dxf']['reason']='unverified_coordinate_frame'
        if rendered['status']=='ready':
            value['dxf']['svg_sha256']=hashlib.sha256(rendered['svg'].encode()).hexdigest()
            value['dxf']['svg_to_canvas']=svg_to_canvas(frames['render_box'],rendered['svg_size'],frames['dxf_to_canvas'])
        return value

    def svg(self,run,page):
        value=self.describe(run,page)
        if value['dxf']['status']!='ready':raise LookupError('DXF comparison not ready')
        key=(str(run['job_id']),run['version'],page,run['manifest']['sha256'],value['dxf']['sha256'],'comparison/1.0')
        with self.lock:rendered=self.jobs[key].result()
        return rendered['svg'].encode(),value
