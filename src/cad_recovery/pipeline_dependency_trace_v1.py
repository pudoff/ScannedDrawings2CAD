"""Python file-open attempts and explicit native consumers for an audit child.

This is observability, not an OS sandbox. Native-library internals and untraced
descendants are explicitly outside the evidence. No inference settings change.
"""
import json
import hashlib
import os
from pathlib import Path
import sys


def read_write(mode, flags=0):
    if isinstance(mode, str):
        return ('r' in mode or '+' in mode), any(c in mode for c in 'wax+')
    return (flags & (os.O_WRONLY | os.O_RDWR)) != os.O_WRONLY, bool(
        flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))


def category(path, generated, source):
    if path == source:
        return 'selected_source'
    if path == generated or path.startswith(generated + '/'):
        return 'this_run'
    if path.startswith('dataset/'):
        return 'other_dataset'
    if '/deps/' in path or path.startswith('.venv/'):
        return 'runtime'
    if path.startswith('artifacts/.cache/'):
        return 'cache'
    if path.startswith('artifacts/'):
        return 'historical_artifact'
    if path.startswith(('src/', 'scripts/', 'configs/')):
        return 'implementation'
    return 'external_or_other'


class Recorder:
    def __init__(self, root, output, source, *, allowed_artifacts=None):
        self.root = os.path.normcase(os.path.abspath(root))
        self.output = Path(output)
        self.generated = self.name(output)
        self.source = self.name(source)
        self.allowed = None if allowed_artifacts is None else set(allowed_artifacts)
        self.events = {}
        self.artifact_bytes = {}
        self.busy = False
        self.fd = os.open(self.output / 'file-attempts.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL)

    def name(self, path):
        path = os.fsdecode(path)
        absolute = os.path.normcase(os.path.abspath(path))
        if absolute.startswith(self.root + os.sep):
            # Preserve display case, compare on the normalized platform path.
            return absolute[len(self.root) + 1:].replace('\\', '/')
        return absolute.replace('\\', '/')

    def caller(self):
        frames = []
        frame = sys._getframe(2)
        integrity = False
        for _ in range(48):
            if frame is None:
                break
            filename = frame.f_code.co_filename
            function = frame.f_code.co_name
            if function == 'file_digest' or (function in ('digest', 'sha256_file')
                                            and 'cad_recovery' in filename):
                integrity = True
            if filename.endswith(('.py', '.pyc')):
                name = self.name(filename)
                if (name.startswith(('src/', 'scripts/')) and
                        not name.endswith('pipeline_dependency_trace_v1.py')):
                    frames.append(f'{name}:{frame.f_lineno}:{function}')
            frame = frame.f_back
        return ('integrity_read' if integrity else 'content_or_import_read'), frames[:12]

    def record(self, path, *, read=True, write=False, consumer='python.open', detail=None):
        if self.busy or isinstance(path, int) or not isinstance(path, (str, bytes, os.PathLike)):
            return
        self.busy = True
        try:
            name = self.name(path)
            scope = category(name, self.generated, self.source)
            purpose, caller = self.caller()
            denied = None
            if read and scope == 'other_dataset':
                denied = 'only_the_selected_dataset_source_is_admitted'
            elif read and scope == 'historical_artifact' and self.allowed is not None and name not in self.allowed:
                denied = 'artifact_outside_observed_allowlist'
            elif write and scope in ('historical_artifact', 'selected_source', 'other_dataset', 'implementation'):
                denied = 'historical_inputs_are_read_only'
            if read and scope == 'historical_artifact' and name not in self.artifact_bytes and not denied:
                try:
                    with open(path, 'rb') as stream:
                        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                    self.artifact_bytes[name] = dict(sha256=sha, size_bytes=os.path.getsize(path))
                except OSError as error:
                    self.artifact_bytes[name] = dict(unavailable=type(error).__name__)
            key = (name, read, write, consumer, purpose, tuple(caller), denied)
            if key not in self.events:
                row = dict(path=name, read=read, write=write, category=scope, purpose=purpose,
                           consumer=consumer, caller=caller, denied=denied, count=0)
                if detail is not None:
                    row['detail'] = detail
                self.events[key] = row
                os.write(self.fd, (json.dumps(row, ensure_ascii=True) + '\n').encode('utf-8'))
            self.events[key]['count'] += 1
            if denied:
                raise PermissionError(f'{denied}: {name}')
        finally:
            self.busy = False

    def hook(self, event, args):
        if event == 'open':
            reading, writing = read_write(args[1], args[2])
            self.record(args[0], read=reading, write=writing)
        elif event == 'subprocess.Popen':
            self.record(args[0], consumer='subprocess.Popen', detail='descendant_not_instrumented')

    def native_consumers(self):
        # Preserve class identity and call arguments. Preloading is part of the
        # measurement overhead; these runs must not replace timing benchmarks.
        import cv2
        import numpy as np
        import onnxruntime as ort
        imread, fromfile = cv2.imread, np.fromfile
        session_init = ort.InferenceSession.__init__
        recorder = self

        def image(filename, *args, **kwargs):
            recorder.record(filename, consumer='cv2.imread')
            return imread(filename, *args, **kwargs)

        def array(file, *args, **kwargs):
            recorder.record(file, consumer='numpy.fromfile')
            return fromfile(file, *args, **kwargs)

        def session(instance, path_or_bytes, *args, **kwargs):
            if isinstance(path_or_bytes, (str, os.PathLike)):
                recorder.record(path_or_bytes, consumer='onnxruntime.InferenceSession')
            return session_init(instance, path_or_bytes, *args, **kwargs)

        cv2.imread, np.fromfile = image, array
        ort.InferenceSession.__init__ = session

    def finish(self, **status):
        self.busy = True
        modules = sorted({self.name(m.__file__) for m in tuple(sys.modules.values())
                          if getattr(m, '__file__', None) and m.__file__.endswith(('.py', '.pyc'))})
        value = dict(schema_version='pipeline-file-trace/1.0', **status,
                     events=list(self.events.values()), imported_modules=modules,
                     historical_artifact_bytes_before_first_attempt=self.artifact_bytes,
                     native_wrappers=['cv2.imread', 'numpy.fromfile', 'onnxruntime.InferenceSession'],
                     limits=['Python open events are attempts, not proof of successful reads',
                             'native-library internals and descendants are not fully traced',
                             'observed inputs are not proof of all future branch dependencies'])
        (self.output / 'trace.json').write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', 'utf-8')
        os.close(self.fd)
