"""On Windows, closing or crashing the worker also kills its processing child."""
import os

class ChildGuard:
    def __init__(self,process):
        self.handle=None
        if os.name!='nt':return
        import ctypes
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_=[('per_process',ctypes.c_int64),('per_job',ctypes.c_int64),('flags',wintypes.DWORD),
                ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',wintypes.DWORD),
                ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
        class IO(ctypes.Structure):_fields_=[(name,ctypes.c_uint64) for name in ('read','write','other','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];kernel.CreateJobObjectW.restype=wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.CreateJobObjectW(None,None)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=Extended();limits.basic.flags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle,wintypes.HANDLE(process._handle)):
            error=ctypes.get_last_error();kernel.CloseHandle(handle);raise ctypes.WinError(error)
        self.kernel=kernel;self.handle=handle

    def close(self):
        if self.handle:self.kernel.CloseHandle(self.handle);self.handle=None
