"""Resource accounting that rejects stale Windows parent PIDs after PID reuse."""
import os
from .service_profiles_v2.budget import AttemptBudget
from .hard_raster_runs_v1 import process_peaks as original_peaks


def lineage(root_pid,root_created,parents,created,job_ids=()):
    """Creation-time ordering is required in addition to a parent PID.

    `created` can be a lazy native lookup. Job membership is authoritative but
    still cannot precede the root's birth in this newly created per-child job.
    """
    selected={root_pid:root_created};rejected=set()
    for pid in job_ids:
        stamp=created(pid)
        if stamp is None:continue
        if stamp<root_created:raise RuntimeError('Job member predates the experiment child')
        selected[pid]=stamp
    while True:
        changed=False
        for pid,parent in parents.items():
            if pid in selected or pid in rejected or parent not in selected:continue
            stamp=created(pid)
            if stamp is None:continue
            if stamp<selected[parent]:rejected.add(pid);continue
            selected[pid]=stamp;changed=True
        if not changed:break
    return selected,rejected


def windows_peaks(process,guard):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype=wintypes.BOOL
    class Pids(ctypes.Structure):
        _fields_=[('assigned',wintypes.DWORD),('count',wintypes.DWORD),('ids',ctypes.c_size_t*1024)]
    ids=Pids()
    if not kernel.QueryInformationJobObject(guard.handle,3,ctypes.byref(ids),ctypes.sizeof(ids),None):
        raise ctypes.WinError(ctypes.get_last_error())
    if ids.count>=1024:raise RuntimeError('Process lineage measurement bound exceeded')
    class Entry(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),('heap',ctypes.c_size_t),
            ('module',wintypes.DWORD),('threads',wintypes.DWORD),('parent',wintypes.DWORD),
            ('priority',wintypes.LONG),('flags',wintypes.DWORD),('name',wintypes.WCHAR*260)]
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    for method in ('Process32FirstW','Process32NextW'):
        getattr(kernel,method).argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)];getattr(kernel,method).restype=wintypes.BOOL
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    parents={}
    try:
        e=Entry();e.size=ctypes.sizeof(e);ok=kernel.Process32FirstW(snapshot,ctypes.byref(e))
        while ok:parents[int(e.pid)]=int(e.parent);ok=kernel.Process32NextW(snapshot,ctypes.byref(e))
    finally:kernel.CloseHandle(snapshot)
    handles={process.pid:(process._handle,False)};stamps={}
    def created(pid):
        if pid in stamps:return stamps[pid]
        if pid not in handles:
            handle=kernel.OpenProcess(0x410,False,pid)
            if not handle:return None  # A process can exit between snapshot and open.
            handles[pid]=(handle,True)
        times=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(wintypes.HANDLE(handles[pid][0]),*[ctypes.byref(t) for t in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        stamps[pid]=times[0].dwLowDateTime+(times[0].dwHighDateTime<<32)
        return stamps[pid]
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),
            *[(n,ctypes.c_size_t) for n in ('peak','working','qpp','qp','qnp','qn','page','peakpage')]]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    try:
        selected,rejected=lineage(process.pid,created(process.pid),parents,created,ids.ids[:ids.count])
        peaks={}
        for pid,stamp in selected.items():
            c=Counters();c.cb=ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(wintypes.HANDLE(handles[pid][0]),ctypes.byref(c),c.cb):
                peaks[(pid,stamp)]=int(c.peak)
        return peaks,len(rejected)
    finally:
        for handle,close in handles.values():
            if close:kernel.CloseHandle(handle)


class LineageBudget(AttemptBudget):
    """The original limits/kill policy, with creation-aware process accounting."""
    def _watch(self):
        self.rejected_stale_parent_links=0
        try:
            while not self.done.wait(.05):
                with self.lock:
                    observed,rejected=windows_peaks(self.process,self.guard) if os.name=='nt' else (original_peaks(self.process,self.guard),0)
                self.rejected_stale_parent_links=max(self.rejected_stale_parent_links,rejected)
                for identity,value in observed.items():self.peaks[identity]=max(self.peaks.get(identity,0),value)
                self.check()
        except Exception as exc:
            self.lineage_failure=str(exc)
            if self.error is None:self.error='budget_monitor_failed'
            self._kill()

    def measurements(self):
        r=super().measurements()
        r.update(process_identity='PID + OS creation time; verified parent order or new job membership',
            rejected_stale_parent_links=getattr(self,'rejected_stale_parent_links',0),
            lineage_failure=getattr(self,'lineage_failure',None),
            process_lifetime_peaks=[{'pid':k[0] if isinstance(k,tuple) else k,'creation_100ns':k[1] if isinstance(k,tuple) else None,
                'peak_bytes':v} for k,v in self.peaks.items()])
        return r
