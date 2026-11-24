"""Reap scratch of dead Windows attempts only; never touch published output."""
from contextlib import contextmanager
import os
from .runtime_io_v1 import read_json, best_effort


@contextmanager
def attempt_lock(folder):
    path = folder/'runtime-owner.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'1');stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def identity_alive(row):
    """Unknown/query-denied is alive; reused PID is not the old process."""
    if os.name != 'nt' or row.get('creation_100ns') is None:
        return True
    import ctypes
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    h=k.OpenProcess(0x1000,False,row['pid'])
    if not h:
        return ctypes.get_last_error()!=87
    try:
        times=[wintypes.FILETIME() for _ in range(4)];exitcode=wintypes.DWORD()
        if not k.GetProcessTimes(h,*[ctypes.byref(t) for t in times]) or not k.GetExitCodeProcess(h,ctypes.byref(exitcode)):
            return True
        stamp=times[0].dwLowDateTime+(times[0].dwHighDateTime<<32)
        return stamp==row['creation_100ns'] and exitcode.value==259
    finally:
        k.CloseHandle(h)


def reap_dead_attempts(files, base):
    from .runtime_worker_v1 import cleanup_private
    base=files.path(base)
    if not base.exists():
        return
    for folder in base.iterdir():
        if not folder.is_dir() or folder.is_symlink() or folder.is_junction():
            continue
        if not (folder/'runtime-owner.lock').exists():
            continue  # Never infer ownership of legacy attempts.
        try:
            with attempt_lock(folder):
                value=read_json(folder/'execution-budget.json')
                peaks=value.get('lifetime_peaks',[])
                if not peaks or any(identity_alive(row) for row in peaks):
                    continue
                cleanup_private(folder)
                best_effort(folder/'orphan-cleanup.json',dict(dead_process_identities_verified=True,
                    private_cache_removed=True,published_output_untouched=True))
        except (OSError,ValueError,KeyError):
            continue  # A live owner, incomplete journal or linked cache is retained.
