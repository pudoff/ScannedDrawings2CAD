"""Immutable uploads and attempt outputs; only database-published manifests are served."""
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
from uuid import uuid4

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write_json(path,value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value,ensure_ascii=True,sort_keys=True,allow_nan=False,indent=2)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())

class Storage:
    def __init__(self,settings):
        self.settings=settings;self.root=settings.root;self.root.mkdir(parents=True,exist_ok=True)

    def path(self,name):
        p=Path(name)
        if p.is_absolute() or not name or '\\' in name or ':' in name or any(x in ('','.','..') for x in name.split('/')):
            raise ValueError('Invalid storage path')
        result=self.root/p
        for ancestor in (result,*result.parents):
            if ancestor==self.root:break
            if ancestor.is_symlink() or getattr(ancestor,'is_junction',lambda:False)():raise ValueError('Linked storage paths are unsupported')
        if not result.resolve().is_relative_to(self.root):raise ValueError('Storage path escapes root')
        return result

    def filename(self,name):
        if not isinstance(name,str) or not 1<=len(name)<=180 or any(x in name for x in '/\\:<>|"?*') or any(ord(x)<32 or ord(x)==127 for x in name):
            raise ValueError('Use a plain filename of at most 180 characters')
        ext=Path(name).suffix.lower()
        if ext not in ('.pdf','.png','.jpg','.jpeg'):raise ValueError('Supported inputs: PDF, PNG, JPEG')
        return ext

    async def upload(self,name,chunks):
        ext=self.filename(name);identifier=str(uuid4());folder=self.path('inputs/'+identifier);folder.mkdir(parents=True)
        path=folder/('source'+ext);size=0;sha=hashlib.sha256();head=b''
        try:
            with path.open('xb') as stream:
                async for chunk in chunks:
                    size+=len(chunk)
                    if size>self.settings.max_upload_bytes:raise OverflowError('Upload size limit exceeded')
                    head=(head+chunk)[:16];sha.update(chunk);stream.write(chunk)
                stream.flush();os.fsync(stream.fileno())
            valid=(ext=='.pdf' and head.startswith(b'%PDF-')) or (ext=='.png' and head.startswith(b'\x89PNG\r\n\x1a\n')) or (ext in ('.jpg','.jpeg') and head.startswith(b'\xff\xd8\xff'))
            if not valid:raise ValueError('File signature does not match supported extension')
        except BaseException:
            path.unlink(missing_ok=True);folder.rmdir();raise
        return {'id':identifier,'original_name':name,'input_path':path.relative_to(self.root).as_posix(),
            'input_sha256':sha.hexdigest(),'size_bytes':size}

    def discard_upload(self,record):
        path=self.path(record['input_path']);path.unlink(missing_ok=True);path.parent.rmdir()

    def attempt(self,run):
        name=f"attempts/{run['id']}/{run['owner']}";path=self.path(name);path.mkdir(parents=True)
        return path

    def output_size(self,folder):
        files=[p for p in folder.rglob('*') if p.is_file()]
        return sum(p.stat().st_size for p in files),len(files)

    def seal(self,folder,summary):
        files=[];total=0
        for path in sorted((folder/'output').rglob('*')):
            if not path.is_file() or '.cache' in path.relative_to(folder).parts:continue
            name=path.relative_to(folder).as_posix();checked=self.path(path.relative_to(self.root).as_posix())
            total+=checked.stat().st_size
            if path.suffix not in ('.png','.json','.dxf'):continue
            files.append({'id':hashlib.sha256(name.encode()).hexdigest()[:24],'path':name,
                'filename':path.name,'sha256':digest(checked),'size_bytes':checked.stat().st_size,
                'media_type':{'.png':'image/png','.json':'application/json','.dxf':'application/dxf'}[path.suffix]})
        if total>self.settings.max_output_bytes or len(files)>self.settings.max_output_files:raise OverflowError('Output resource limit exceeded')
        if not files or len({f['id'] for f in files})!=len(files):raise ValueError('Missing or repeated result artifacts')
        target=folder/'result-manifest.json';write_json(target,{'schema_version':'service-result/1.0','summary':summary,'files':files})
        return {'path':target.relative_to(self.root).as_posix(),'sha256':digest(target)}

    def manifest(self,run):
        if run['status']!='completed' or not run['manifest']:raise LookupError('Result is not completed')
        path=self.path(run['manifest']['path'])
        if digest(path)!=run['manifest']['sha256']:raise ValueError('Published manifest changed')
        return path,json.loads(path.read_text('utf-8'))

    def artifact(self,run,identifier):
        path,manifest=self.manifest(run)
        entry=next((f for f in manifest['files'] if f['id']==identifier),None)
        if entry is None:raise KeyError(identifier)
        candidate=path.parent/entry['path'];candidate=self.path(candidate.relative_to(self.root).as_posix())
        if not candidate.resolve().is_relative_to(path.parent.resolve()) or digest(candidate)!=entry['sha256'] or candidate.stat().st_size!=entry['size_bytes']:
            raise ValueError('Published artifact changed')
        return candidate,entry
