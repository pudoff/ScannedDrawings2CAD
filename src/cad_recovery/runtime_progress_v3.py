"""Exception-safe V2 phase accounting with the common diagnostic writer."""
from .runtime_progress_v2 import Progress as Previous
from .runtime_io_v1 import atomic_json


class Progress(Previous):
    def snapshot(self):
        result = super().snapshot()
        result['writer']['version'] = 'runtime-progress/3.0'
        return result

    def write(self):
        with self.lock:
            self.sequence += 1
            self.replace_retries += atomic_json(self.path, self.snapshot())
