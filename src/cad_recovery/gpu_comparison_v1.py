"""Frozen acceptance and memory observation for the paired OCR experiment."""
import ctypes
from ctypes import wintypes
import statistics
import threading

POLICY = {
    'version': 'paired-fp32-ocr/1.0', 'atol': 1e-4, 'rtol': 1e-3,
    'warm_repeats': 3, 'numeric_cases': ['jpeg', 'train-2', 'validation-1-a'],
    'warm_validation_speedup_min': 1.10, 'cold_validation_slowdown_max': 1.10,
    'text_field_regressions_allowed': 0, 'symbol_losses_allowed': 0,
    'association_candidate_set_changes_allowed': 0,
    'cpu_anchor_text_or_polygon_changes_allowed': 0, 'polygon_atol_px': .01,
    'timeout_seconds': 300, 'memory_MiB': 4096, 'whole_GPU_memory_ceiling_MiB': 12288,
    'model_dtype': 'FP32', 'tf32': False, 'closed_tests_executed': False,
    'failures': 'retain assigned tasks; missing evidence prevents acceptance',
    'timing': 'instrumented observation route: intake, preprocessing, geometry, OCR, arrows, associations, empty-graph DXF export',
    'cold': 'fresh process plus model initialization and first route; OS file cache uncontrolled; shared project CUDA JIT cache',
    'warm': 'three complete routes with reused sessions; median per case; paired medians on validation only',
    'vram': 'NVML whole-device used memory at 50 ms; includes desktop/other applications; not attributable per-process VRAM under WDDM',
    'adoption': 'eligible for explicit GPU observation profile only; no automatic change of production CPU default',
}


def decide(rows):
    validations = [r for r in rows if r['split'] == 'validation']
    warm_speedup = statistics.median(r['warm_cpu']/r['warm_cuda'] for r in validations) if validations else None
    cold_slowdown = statistics.median(r['cold_cuda']/r['cold_cpu'] for r in validations) if validations else None
    quality = all(r['quality_pass'] for r in rows)
    numeric = all(r['numeric_pass'] for r in rows if r['id'] in POLICY['numeric_cases'])
    complete = bool(rows) and bool(validations) and {r['id'] for r in rows} >= set(POLICY['numeric_cases']) and all(r['complete'] for r in rows)
    performance = warm_speedup is not None and warm_speedup >= POLICY['warm_validation_speedup_min'] and cold_slowdown <= POLICY['cold_validation_slowdown_max']
    return {'eligible_for_explicit_gpu_profile': complete and quality and numeric and performance,
            'complete': complete, 'quality_pass': quality, 'numeric_pass': numeric,
            'performance_pass': performance, 'median_validation_warm_speedup': warm_speedup,
            'median_validation_cold_slowdown': cold_slowdown, 'production_default': 'CPU',
            'quality_improvement_claimed': False}


class DeviceMemorySampler:
    """Whole-device VRAM, deliberately separate from process-tree host RAM."""
    def __init__(self):
        self.library = ctypes.WinDLL('nvml.dll')
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError('NVML initialization failed')
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError('NVML device unavailable')
        self.stop = threading.Event()
        self.values = []
        self.errors = []

    def sample(self):
        class Memory(ctypes.Structure):
            _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
        memory = Memory()
        status = self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory))
        if status:
            self.errors.append(status)
        else:
            self.values.append(memory.used/1024**2)

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(.05):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()
        self.library.nvmlShutdown()

    def report(self):
        return {'scope': POLICY['vram'], 'before_MiB': self.values[0] if self.values else None,
                'peak_MiB': max(self.values) if self.values else None,
                'peak_minus_before_MiB': max(self.values)-self.values[0] if self.values else None,
                'sample_count': len(self.values), 'errors': self.errors}
