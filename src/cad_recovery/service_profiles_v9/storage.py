"""Budget scans tolerate a bounded rename while a producer archives an export."""
import threading
from stat import S_ISREG
from cad_recovery.service_profiles_v2.storage import Storage as PreviousStorage


class Storage(PreviousStorage):
    def __init__(self,settings):
        super().__init__(settings)
        self._scan_lock=threading.Lock()

    def output_size(self,folder):
        # Producers replace/rename files. A separate is_file()/stat() pair can
        # otherwise abort a healthy child mid-upgrade. Rescan the whole tree;
        # do not silently skip vanished files or catch permission/I/O errors.
        with self._scan_lock:
            for attempt in range(4):
                try:
                    sizes=[]
                    for path in folder.rglob('*'):
                        value=path.stat()
                        if S_ISREG(value.st_mode):sizes.append(value.st_size)
                    return sum(sizes),len(sizes)
                except FileNotFoundError:
                    if attempt==3:raise
