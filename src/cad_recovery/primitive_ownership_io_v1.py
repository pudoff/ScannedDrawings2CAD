"""Semantic sidecars refer to immutable geometry; they never replace a graph."""
from copy import deepcopy
from pathlib import Path
import json
from PIL import Image
import numpy as np
from .dataset_layout_v1 import checked,pin
from .geometry_bundle_io_v2 import load as load_bundle
from .geometry_bundle_io_v1 import write
from .primitive_ownership_v1 import analyze
from .part_admission_v1 import object_sha


def producer(root,contract):
    data=json.loads(checked(root,contract).read_text(encoding='utf-8'))
    required={'src/cad_recovery/primitive_ownership_v1.py','src/cad_recovery/primitive_ownership_io_v1.py'}
    if not required<={p['path'] for p in data['code']}:raise ValueError('Incomplete ownership producer binding')
    for ref in data['code']:checked(root,ref)
    return deepcopy(contract)


def capture(root,record,view,folder,*,bundle_model,producer_contract):
    root=Path(root).resolve();folder=Path(folder).resolve();folder.relative_to(root)
    if folder.exists():raise FileExistsError(folder)
    bound=producer(root,producer_contract)
    packet=load_bundle(root,record,view,model_contract=bundle_model)
    gray=np.array(Image.open(checked(root,view['gray'])).convert('L'))
    result=analyze(gray,packet)
    descriptor={'source_bundle':deepcopy(record),'view':{k:deepcopy(view[k]) for k in ('gray','input_to_native')},
        'bundle_model':deepcopy(bundle_model),'producer':bound,'result':result}
    descriptor['content_sha256']=object_sha(descriptor);folder.mkdir(parents=True)
    write(folder/'ownership.json',descriptor)
    return pin(root,(folder/'ownership.json').relative_to(root).as_posix())


def load(root,reference,record,view,*,bundle_model,producer_contract):
    root=Path(root).resolve();d=json.loads(checked(root,reference).read_text(encoding='utf-8'))
    expected=d.pop('content_sha256')
    if expected!=object_sha(d):raise ValueError('Ownership content SHA differs')
    if d['source_bundle']!=record or d['view']!={k:view[k] for k in ('gray','input_to_native')}:
        raise ValueError('Ownership input/frame binding differs')
    if d['bundle_model']!=bundle_model or d['producer']!=producer(root,producer_contract):raise ValueError('Ownership model differs')
    packet=load_bundle(root,record,view,model_contract=bundle_model)
    gray=np.array(Image.open(checked(root,view['gray'])).convert('L'));result=analyze(gray,packet)
    if object_sha(result)!=object_sha(d['result']):raise ValueError('Ownership evidence differs')
    if not result['complete']:raise ValueError('Incomplete ownership cannot be handed off')
    return result,packet
