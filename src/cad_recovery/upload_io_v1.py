"""One awaited disk operation per upload; cancellation joins it before cleanup."""
import asyncio
import hashlib
import os
from uuid import uuid4

WRITE_WINDOW=256*1024


async def completed_io(function,*args,propagate_cancel=True):
    """A cancelled await does not stop its thread or transfer file ownership.

    Keep waiting, including after repeated cancellation, before another
    operation may touch the same stream. No filesystem work runs in the loop.
    """
    task=asyncio.create_task(asyncio.to_thread(function,*args))
    cancellation=None
    while not task.done():
        try:await asyncio.shield(task)
        except asyncio.CancelledError as error:cancellation=error
        except BaseException:break
    if cancellation is not None and propagate_cancel:
        # Consume the thread exception as well; the caller still owns cleanup.
        try:task.result()
        except BaseException:pass
        raise cancellation
    return task.result()


class Upload:
    """All methods, including path validation and removal, run in a thread."""
    def __init__(self,files,name):
        self.files=files;self.name=name;self.stream=None;self.created=False
        self.folder=self.path=None;self.size=0;self.sha=hashlib.sha256();self.head=b''

    def open(self):
        self.ext=self.files.filename(self.name);self.identifier=str(uuid4())
        self.folder=self.files.path('inputs/'+self.identifier)
        self.folder.mkdir(parents=True);self.created=True
        self.path=self.folder/('source'+self.ext)
        self.stream=self.path.open('xb')

    def write(self,chunk):
        size=self.size+len(chunk)
        if size>self.files.settings.max_upload_bytes:raise OverflowError('Upload limit exceeded')
        view=memoryview(chunk)
        for start in range(0,len(view),WRITE_WINDOW):
            part=view[start:start+WRITE_WINDOW]
            while part:
                count=self.stream.write(part)
                if not isinstance(count,int) or count<=0 or count>len(part):raise OSError('Incomplete upload write')
                part=part[count:]
        self.size=size;self.sha.update(view);self.head=(self.head+bytes(view[:16]))[:16]

    def finish(self):
        valid=(self.ext=='.pdf' and self.head.startswith(b'%PDF-')) or (
            self.ext=='.png' and self.head.startswith(b'\x89PNG\r\n\x1a\n')) or (
            self.ext in ('.jpg','.jpeg') and self.head.startswith(b'\xff\xd8\xff'))
        if not valid:raise ValueError('File signature differs from extension')
        self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
        return dict(id=self.identifier,original_name=self.name,
            input_path=self.path.relative_to(self.files.root).as_posix(),
            input_sha256=self.sha.hexdigest(),size_bytes=self.size)

    def abort(self):
        # There is no in-flight writer now. Try every cleanup step even if one
        # fails; an undeletable orphan is never acknowledged or enqueued here.
        errors=[]
        if self.stream is not None:
            try:self.stream.close()
            except OSError as error:errors.append(error)
            self.stream=None
        if self.created:
            if self.path is not None:
                try:self.path.unlink(missing_ok=True)
                except OSError as error:errors.append(error)
            try:self.folder.rmdir()
            except OSError as error:errors.append(error)
        if errors:raise errors[0]


async def upload(files,name,chunks):
    state=Upload(files,name)
    try:
        await completed_io(state.open)
        async for chunk in chunks:
            # Do not request another body chunk until this write/hash finishes.
            await completed_io(state.write,chunk)
        return await completed_io(state.finish)
    except BaseException:
        await completed_io(state.abort,propagate_cancel=False)
        raise
