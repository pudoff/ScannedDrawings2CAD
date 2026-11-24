"""Bounded asynchronous uploads and sealed nested evidence inventory."""
import asyncio
import hashlib
import os
from uuid import uuid4
from cad_recovery.service_v1.storage import Storage as HistoricalStorage,write_json,digest
from cad_recovery.family_catalog_v2 import require

class Storage(HistoricalStorage):
    async def upload(self,name,chunks):
        ext=self.filename(name);identifier=str(uuid4());folder=self.path('inputs/'+identifier)
        await asyncio.to_thread(folder.mkdir,parents=True)
        path=folder/('source'+ext);stream=await asyncio.to_thread(path.open,'xb');size=0;sha=hashlib.sha256();head=b''
        def write(chunk):stream.write(chunk);sha.update(chunk)
        try:
            async for chunk in chunks:
                size+=len(chunk)
                if size>self.settings.max_upload_bytes:raise OverflowError('Upload limit exceeded')
                head=(head+chunk[:16])[:16]
                task=asyncio.create_task(asyncio.to_thread(write,chunk))
                try:await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            await asyncio.to_thread(stream.flush);await asyncio.to_thread(os.fsync,stream.fileno());await asyncio.to_thread(stream.close)
            valid=(ext=='.pdf' and head.startswith(b'%PDF-')) or (ext=='.png' and head.startswith(b'\x89PNG\r\n\x1a\n')) or (ext in ('.jpg','.jpeg') and head.startswith(b'\xff\xd8\xff'))
            require(valid,'File signature differs from extension')
        except BaseException:
            await asyncio.shield(asyncio.to_thread(stream.close))
            await asyncio.shield(asyncio.to_thread(path.unlink,missing_ok=True));await asyncio.shield(asyncio.to_thread(folder.rmdir));raise
        return {'id':identifier,'original_name':name,'input_path':path.relative_to(self.root).as_posix(),'input_sha256':sha.hexdigest(),'size_bytes':size}

    def seal(self,folder,summary):
        files=[];inventory=[]
        for path in sorted((folder/'output').rglob('*')):
            if not path.is_file():continue
            checked=self.path(path.relative_to(self.root).as_posix());name=checked.relative_to(folder).as_posix()
            item={'path':name,'sha256':digest(checked),'size_bytes':checked.stat().st_size};inventory.append(item)
            if path.parent==folder/'output' and path.suffix in ('.png','.json','.dxf'):
                files.append({**item,'id':hashlib.sha256(name.encode()).hexdigest()[:24],'filename':path.name,
                    'media_type':{'.png':'image/png','.json':'application/json','.dxf':'application/dxf'}[path.suffix]})
        require(bool(files),'No published artifacts')
        evidence=folder/'output/evidence-inventory.json';write_json(evidence,{'files':inventory})
        files.append({'path':'output/evidence-inventory.json','sha256':digest(evidence),'size_bytes':evidence.stat().st_size,
            'id':hashlib.sha256(b'output/evidence-inventory.json').hexdigest()[:24],'filename':evidence.name,'media_type':'application/json'})
        target=folder/'result-manifest.json';write_json(target,{'schema_version':'service-result/2.0','summary':summary,
            'files':files,'inventory':inventory+[{'path':'output/evidence-inventory.json','sha256':digest(evidence),'size_bytes':evidence.stat().st_size}]})
        return {'path':target.relative_to(self.root).as_posix(),'sha256':digest(target)}

    def manifest(self,run):
        path,manifest=super().manifest(run)
        if manifest.get('schema_version')=='service-result/2.0':
            inventory=manifest['inventory'];actual={p.relative_to(path.parent).as_posix() for p in (path.parent/'output').rglob('*') if p.is_file()}
            require(actual=={r['path'] for r in inventory},'Sealed artifact inventory changed')
            for record in inventory:
                p=self.path((path.parent/record['path']).relative_to(self.root).as_posix())
                require(p.stat().st_size==record['size_bytes'] and digest(p)==record['sha256'],'Sealed evidence changed')
        return path,manifest
