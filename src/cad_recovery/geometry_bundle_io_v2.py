"""Checked experimental codec; reuse v1 serialization and reference inspection."""
from copy import deepcopy
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked,pin
from .geometry_bundle_io_v1 import FILES,IMAGES,write,model_binding as previous_binding
from .geometry_bundle_v1 import inspect,matrix
from .geometry_bundle_v2 import select,POLICY
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import foreground


def model_binding(root,contract):
    models=previous_binding(root,contract)
    code=json.loads(checked(root,contract).read_text(encoding='utf-8'))['code']
    required={'src/cad_recovery/'+s+'.py' for s in ('robust_topology_v2','observed_geometry_qa_v2','geometry_bundle_v2','geometry_bundle_io_v2')}
    if not required<={p['path'] for p in code}:raise ValueError('Missing experimental producer code binding')
    return models


def save(root,view,folder,packet,*,model_contract):
    root=Path(root).resolve();folder=Path(folder).resolve();folder.relative_to(root)
    if folder.exists():raise FileExistsError(folder)
    if packet['models']!=model_binding(root,model_contract):raise ValueError('Unbound experimental model')
    gray=np.array(Image.open(checked(root,view['gray'])).convert('L'))
    if sha(gray)!=packet['input_sha256'] or not inspect(gray,packet,view['input_to_native'])['valid']:
        raise ValueError('Invalid observation binding')
    folder.mkdir(parents=True)
    Image.fromarray(gray).save(folder/'input.png');refs={'input':pin(root,(folder/'input.png').relative_to(root).as_posix())}
    for key,name in FILES.items():
        write(folder/name,packet[key]);refs[key]=pin(root,(folder/name).relative_to(root).as_posix())
    for key,name in IMAGES.items():
        Image.fromarray(packet[key]).save(folder/name);refs[key]=pin(root,(folder/name).relative_to(root).as_posix())
    keys=('revision','prefix','input_sha256','input_size_px','input_to_native','native_to_input','models','selection','policy',
        'inspection','complete','dependency_state','dependent_graph_rebuilt','cached_dependent_inputs_used',
        'source_pixels_erased','production_changed','physical_geometry_approved','source_ownership_work')
    descriptor={k:deepcopy(packet[k]) for k in keys}
    descriptor.update(schema='geometry-bundle/2.0',input_file=deepcopy(view['gray']),files=refs,
        owner_interval_ids=[p['id'] for p in packet['graph']['proposals']],observations_require_review=True)
    descriptor['content_sha256']=object_sha(descriptor);write(folder/'bundle.json',descriptor)
    return pin(root,(folder/'bundle.json').relative_to(root).as_posix())


def load(root,record,view,*,model_contract):
    root=Path(root).resolve();descriptor=json.loads(checked(root,record).read_text(encoding='utf-8'))
    content=descriptor.pop('content_sha256')
    if content!=object_sha(descriptor) or descriptor['schema']!='geometry-bundle/2.0':raise ValueError('Invalid experimental manifest')
    if descriptor['models']!=model_binding(root,model_contract) or descriptor['policy']!=POLICY:raise ValueError('Experimental model differs')
    if descriptor['input_file']!=view['gray']:raise ValueError('Input file differs')
    m=matrix(view['input_to_native'])
    if not np.array_equal(m,descriptor['input_to_native']) or not np.array_equal(np.linalg.inv(m),descriptor['native_to_input']):
        raise ValueError('Coordinate frame differs')
    gray=np.array(Image.open(checked(root,view['gray'])).convert('L'));refs=descriptor['files']
    for value in refs.values():checked(root,value)
    if descriptor['input_size_px']!=list(gray.shape[::-1]) or descriptor['input_sha256']!=sha(gray):raise ValueError('Input pixels differ')
    if not np.array_equal(gray,np.array(Image.open(checked(root,refs['input'])).convert('L'))):raise ValueError('Saved pixels differ')
    packet={**descriptor,**{k:json.loads(checked(root,refs[k]).read_text(encoding='utf-8')) for k in FILES}}
    for k in IMAGES:packet[k]=np.array(Image.open(checked(root,refs[k])))
    for field in ('source_owner','role_owner'):
        a=packet[field]
        if not np.issubdtype(a.dtype,np.integer) or a.min(initial=0)<0 or a.max(initial=0)>65535:raise ValueError('Invalid owner values')
        packet[field]=a.astype(np.uint16)
    selection=select(gray,packet['raw_source'],packet['proposed_geometry'],input_to_native=m,models=packet['models'],
        extraction_complete=packet['selection']['extraction_complete'])
    for k in ('revision','prefix','source','geometry','selection'):
        if object_sha(selection[k])!=object_sha(packet[k]):raise ValueError('Selected geometry differs: '+k)
    if descriptor['owner_interval_ids']!=[p['id'] for p in packet['graph']['proposals']]:raise ValueError('Owner references differ')
    inspection=inspect(gray,packet,m)
    if not inspection['valid'] or inspection!=descriptor['inspection']:raise ValueError('Rebuilt references differ')
    if not descriptor['dependent_graph_rebuilt'] or descriptor['cached_dependent_inputs_used']:raise ValueError('Stale dependencies')
    if descriptor['complete']!=all(descriptor['dependency_state'].values()):raise ValueError('Incomplete search reported complete')
    raw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    excluded=np.where((raw>0)&(foreground(gray)==0),255,0).astype(np.uint8)
    if not np.array_equal(raw,packet['raw_ink']) or not np.array_equal(excluded,packet['preprocessing_excluded_ink']):raise ValueError('Source ink differs')
    if descriptor['physical_geometry_approved'] or descriptor['production_changed']:raise ValueError('Unreviewed promotion')
    return packet
