"""Authenticated, bounded derivation from the published manifest only."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import encoded
from cad_recovery.observation_graph_v1 import build,build_flat,matrix,VERSION


class InspectionLimit(ValueError):pass


class Inspector:
    def __init__(self,storage):
        self.files=storage;self.cache=OrderedDict();self.cache_bytes=0;self.lock=Lock()

    def _json(self,root,entries,name):
        item=next((e for e in entries if e['filename']==name),None)
        if item is None:raise KeyError(name)
        if item['size_bytes']>32*1024**2:raise InspectionLimit('Observation JSON exceeds inspection limit')
        path=self.files.path((root/item['path']).relative_to(self.files.root).as_posix())
        data=path.read_bytes()
        if len(data)!=item['size_bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Observation artifact changed')
        return json.loads(data,parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

    def inspect(self,run,page_index):
        # Validate the entire sealed inventory even when a derived value is cached.
        manifest_path,manifest=self.files.manifest(run)
        summary=manifest['summary'];count=summary.get('page_count',0)
        if not 0<=page_index<count:raise KeyError('Page not found')
        key=(run['id'],str(run['job_id']),run['version'],page_index,run['manifest']['sha256'],run.get('input_sha256'),VERSION)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return json.loads(self.cache[key])
        root=manifest_path.parent;entries=manifest['files'];prefix=f'page-{page_index+1:03d}'
        report=self._json(root,entries,'report.json')
        if run.get('input_sha256') and report.get('source_sha256')!=run['input_sha256']:
            raise ValueError('Published result belongs to another input')
        native=next((e for e in entries if e['filename']==prefix+'-observations.json'),None)
        if native:
            source=self._json(root,entries,native['filename'])
            if source.get('page_index',page_index)!=page_index:raise ValueError('Observation page mismatch')
            source['page_index']=page_index
            envelope=build(source)
            chain=source.get('coordinate_chain',{});to_analysis=matrix(chain.get('original_to_analysis',np.eye(3)))
            if chain.get('page_index',page_index)!=page_index:raise ValueError('Observation page frame mismatch')
            analysis_size=chain.get('analysis_size_px',source['source_size_px'])
            if chain.get('original_sha256',source['source_sha256'])!=source['source_sha256']:
                raise ValueError('Observation original frame source mismatch')
            if chain.get('source_pdf_sha256',report['source_sha256'])!=report['source_sha256']:
                raise ValueError('Observation PDF source mismatch')
            preview_name=prefix+'-source.png';observed_sha=native['sha256']
        else:
            geo_name=prefix+'-geometry.json'
            if not any(e['filename']==geo_name for e in entries):
                raise LookupError('This version has no raster observation reports; select native analysis or presentation analysis')
            geometry=self._json(root,entries,geo_name)
            common={k:geometry[k] for k in ('source_sha256','processed_size_px')}
            def optional(name,empty):
                return self._json(root,entries,name) if any(e['filename']==name for e in entries) else {**common,**empty}
            ocr=optional(prefix+'-ocr.json',{'lines':[],'notations':[]})
            arrows=optional(prefix+'-arrows.json',{'candidates':[]})
            associations=optional(prefix+'-dimension-associations.json',{'associations':[]})
            envelope=build_flat(geometry=geometry,ocr=ocr,arrows=arrows,associations=associations,page_index=page_index)
            analysis_size=common['processed_size_px'];to_analysis=np.eye(3)
            preview_name=prefix+'-processed.png'
            observed_sha=hashlib.sha256(encoded([e['sha256'] for e in entries if e['filename'] in
                (geo_name,prefix+'-ocr.json',prefix+'-arrows.json',prefix+'-dimension-associations.json')])).hexdigest()
        preview=next((e for e in entries if e['filename']==preview_name),None)
        if preview is None:raise KeyError('Observation frame preview missing')
        path=self.files.path((root/preview['path']).relative_to(self.files.root).as_posix())
        if path.stat().st_size!=preview['size_bytes']:raise ValueError('Preview changed')
        with Image.open(path) as image:size=list(image.size)
        if len(analysis_size)!=2 or any(type(n)is not int or n<=0 for n in analysis_size):
            raise ValueError('Invalid analysis frame dimensions')
        scale=np.diag([size[0]/analysis_size[0],size[1]/analysis_size[1],1.])
        envelope['preview']={'artifact_id':preview['id'],'size_px':size,
                             'observation_to_preview':(scale@to_analysis).tolist()}
        envelope['context']={'job_id':str(run['job_id']),'version':run['version'],'page_index':page_index,
            'manifest_sha256':run['manifest']['sha256'],'observation_sha256':observed_sha,
            'input_sha256':report.get('source_sha256'),
            'profile':run['options']['profile'],'adapter_version':VERSION}
        data=encoded(envelope)
        if len(data)>16*1024**2:raise InspectionLimit('Derived observation graph exceeds inspection limit')
        with self.lock:
            if key not in self.cache:
                self.cache[key]=data;self.cache_bytes+=len(data)
                while self.cache_bytes>32*1024**2 or len(self.cache)>8:
                    _,old=self.cache.popitem(last=False);self.cache_bytes-=len(old)
        return json.loads(data)
