"""Explicit experimental CPU/CUDA adapters with the pinned RapidOCR pipeline.

The production CPU class and its distribution/version guard are unchanged.
Only its public recognize/provenance interface is inherited here.
"""
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path

import numpy as np

from .gpu_inference_v1 import local_models, sha256, summarize_profile
from .ocr_engine import LocalOCREngine


def rapid_parameters(models: list[dict], root: Path) -> dict:
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion
    params = {
        'Global.text_score': 0.0, 'Global.use_det': True, 'Global.use_cls': True,
        'Global.use_rec': True, 'Global.log_level': 'warning', 'Global.max_side_len': 2000,
        'Global.return_word_box': False, 'Global.return_single_char_box': False,
        'Global.model_root_dir': str(root/'artifacts/models/ppocr-v5'),
        'Det.limit_type': 'max', 'Det.limit_side_len': 2000,
        'Det.lang_type': LangDet.CH, 'Cls.lang_type': LangCls.CH, 'Rec.lang_type': LangRec.ESLAV,
        'Rec.rec_keys_path': None, 'Rec.rec_batch_num': 6,
        'EngineConfig.onnxruntime.intra_op_num_threads': min(4, os.cpu_count() or 1),
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        'EngineConfig.onnxruntime.use_cuda': False, 'EngineConfig.onnxruntime.use_dml': False,
        'EngineConfig.onnxruntime.use_cann': False, 'EngineConfig.onnxruntime.use_coreml': False,
    }
    for row in models:
        section = row['role'].capitalize()
        params.update({section+'.model_path': str(root/row['path']), section+'.engine_type': EngineType.ONNXRUNTIME,
                       section+'.model_type': ModelType.MOBILE, section+'.ocr_version': OCRVersion.PPOCRV5})
    return params


class RecordedSession:
    """Record the first real normalized input/output without changing inference."""
    def __init__(self, session, directory: Path, role: str, capture: bool):
        self.session = session
        self.directory = directory
        self.role = role
        self.capture = capture
        self.recorded = False

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, feed, *args, **kwargs):
        outputs = self.session.run(output_names, feed, *args, **kwargs)
        if self.capture and not self.recorded:
            folder = self.directory/self.role
            folder.mkdir(parents=True)
            if len(feed) != 1:
                raise RuntimeError('Expected one normalized model input')
            np.save(folder/'input.npy', next(iter(feed.values())), allow_pickle=False)
            for index, array in enumerate(outputs):
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise RuntimeError('Nonfinite or non-FP32 neural output')
                np.save(folder/('output-'+str(index)+'.npy'), array, allow_pickle=False)
            self.recorded = True
        return outputs


class PairedOCREngine(LocalOCREngine):
    def __init__(self, root: Path, device: str, evidence: Path, *, capture=False):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Explicit cpu/cuda device is required')
        if version('onnxruntime-gpu') != '1.31.0' or version('rapidocr') != '3.9.2':
            raise RuntimeError('The experimental adapter requires the pinned GPU environment')
        try:
            version('onnxruntime')
        except PackageNotFoundError:
            pass
        else:
            raise RuntimeError('CPU and GPU ORT distributions cannot coexist')
        import onnxruntime as ort
        from rapidocr import RapidOCR
        from rapidocr.utils.parse_parameters import ParseParams
        from omegaconf import flag_override
        models = local_models(root)
        evidence.mkdir(parents=True, exist_ok=False)
        ort.set_default_logger_severity(3)
        if device == 'cuda':
            ort.preload_dlls(directory='')
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA provider unavailable')
        self.sessions = {}
        params = rapid_parameters(models, root)
        for row in models:
            role = row['role']
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, os.cpu_count() or 1)
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_profiling = True
            options.profile_file_prefix = str(evidence/role)
            providers = ['CPUExecutionProvider'] if device == 'cpu' else [
                ('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0, 'gpu_mem_limit': 4*1024**3,
                 'cudnn_conv_algo_search': 'HEURISTIC', 'do_copy_in_default_stream': 1}), 'CPUExecutionProvider']
            session = ort.InferenceSession(str(root/row['path']), sess_options=options, providers=providers)
            session.disable_fallback()
            expected = 'CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'
            if session.get_providers()[0] != expected:
                raise RuntimeError('Unexpected session fallback: '+role)
            if device == 'cuda' and session.get_provider_options()['CUDAExecutionProvider'].get('use_tf32') != '0':
                raise RuntimeError('TF32 is not disabled')
            self.sessions[role] = session
            params[role.capitalize()+'.session'] = session

        class SessionsRapidOCR(RapidOCR):
            # The pinned upstream explicitly supports provided ONNX sessions.
            # Its OmegaConf loader needs allow_objects for those session values.
            def _load_config(self, config_path, supplied):
                cfg = super()._load_config(config_path, None)
                with flag_override(cfg, 'allow_objects', True):
                    return ParseParams.update_batch(cfg, supplied)

        self._backend = SessionsRapidOCR(params=params)
        for role in self.sessions:
            component = getattr(self._backend, 'text_'+role)
            component.session.session = RecordedSession(self.sessions[role], evidence/'tensors', role, capture)
        self._provenance = {'engine': 'RapidOCR', 'engine_version': '3.9.2', 'onnxruntime_version': '1.31.0',
                            'distribution': 'onnxruntime-gpu', 'provider': expected, 'device': device,
                            'models': models, 'offline': True, 'tf32': False,
                            'preprocessing': 'pinned RapidOCR 3.9.2, production parameters',
                            'experimental': True, 'production_enabled': False}
        self.backend = self._backend
        self.evidence = evidence
        self.device = device

    def recognize_line(self, bgr: np.ndarray) -> str:
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        output = self.backend.text_rec(TextRecInput(img=np.array(bgr, copy=True, order='C'), return_word_box=False))
        if len(output.txts) != 1:
            raise RuntimeError('Expected one recognition-only line')
        return output.txts[0]

    def replay_tensors(self, source: Path, policy: dict) -> list[dict]:
        rows = []
        for role, session in self.sessions.items():
            folder = source/role
            if not (folder/'input.npy').exists():
                rows.append({'role': role, 'pass': False, 'reason': 'assigned_tensor_missing'})
                continue
            tensor = np.load(folder/'input.npy', allow_pickle=False)
            outputs = session.run(None, {session.get_inputs()[0].name: tensor})
            for index, value in enumerate(outputs):
                reference = np.load(folder/('output-'+str(index)+'.npy'), allow_pickle=False)
                same_shape = value.shape == reference.shape
                finite = bool(value.dtype == reference.dtype == np.float32 and np.isfinite(value).all() and np.isfinite(reference).all())
                close = same_shape and finite and bool(np.allclose(reference, value, atol=policy['atol'], rtol=policy['rtol']))
                difference = np.abs(reference-value) if same_shape and finite else None
                rows.append({'role': role, 'output': index, 'pass': close, 'finite': finite,
                             'shape': list(value.shape), 'reference_input_sha256': sha256(folder/'input.npy'),
                             'max_absolute_difference': float(difference.max()) if difference is not None else None,
                             'argmax_equal_fraction': float(np.mean(np.argmax(reference, axis=-1) == np.argmax(value, axis=-1))) if same_shape else None})
        return rows

    def finish_profiles(self) -> dict:
        result = {}
        for role, session in self.sessions.items():
            path = Path(session.end_profiling())
            events = json.loads(path.read_text('utf-8'))
            if self.device == 'cuda':
                distribution = summarize_profile(events)
            else:
                from collections import Counter
                counts = Counter(e.get('args', {}).get('provider') for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
                if set(counts) != {'CPUExecutionProvider'}:
                    raise RuntimeError('Unexpected non-CPU execution in the CPU arm')
                distribution = {'executed_node_events': dict(counts)}
            result[role] = {'path': str(path), 'placement': distribution}
        return result
