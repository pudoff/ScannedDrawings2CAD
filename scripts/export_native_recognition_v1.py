"""ABI-compatible isolated exporter; no customer samples or labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.ocr_recognition_core_v1 import native_environment,native_model


def digest(path):
    with path.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert digest(a.config)==a.config_sha256;c=json.loads(a.config.read_text('utf-8'))
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'))
    native_environment(ROOT,runtime['source_root'])
    os.environ['POLYGRAPHY_AUTOINSTALL_DEPS']='0'
    import paddle
    import paddle2onnx
    import onnx
    import onnxruntime as ort
    import numpy as np
    paddle.set_device('cpu');paddle.seed(20261108)
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    target=a.output.resolve();assert target.is_relative_to(ROOT/'artifacts') and not target.exists();target.mkdir()
    temporary=target/'temporary';temporary.mkdir();tempfile.tempdir=str(temporary)
    records=[]
    for name,checkpoint in c['export_cases'].items():
        local={**runtime,'checkpoint':checkpoint['path']};tick=time.perf_counter();model,chars,_=native_model(ROOT,local)
        case_directory=temporary/name;case_directory.mkdir()
        paddle.jit.save(model,str(case_directory/'model'),input_spec=[paddle.static.InputSpec([None,3,48,320],'float32','image')])
        # paddle.onnx.export ignores these options; call the converter explicitly.
        paddle2onnx.export(str(case_directory/'model.json'),str(case_directory/'model.pdiparams'),
                          save_file=str(target/(name+'.onnx')),opset_version=17,
                          enable_optimize=False,optimize_tool=None)
        path=target/(name+'.onnx');onnx.checker.check_model(onnx.load(str(path)))
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);cases=[]
        for n in (1,2):
            x=np.random.default_rng(20261108).uniform(-1,1,(n,3,48,320)).astype('float32')
            with paddle.no_grad():reference=model(paddle.to_tensor(x)).numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x})[0]
            np.testing.assert_allclose(reference,actual,rtol=0.0001,atol=0.00001)
            np.save(target/f'{name}-batch-{n}.npy',actual,allow_pickle=False)
            cases.append({'batch':n,'shape':list(actual.shape),'max_abs':float(np.abs(reference-actual).max())})
        records.append({'name':name,'checkpoint':checkpoint,'onnx_sha256':digest(path),'synthetic_cases':cases,'seconds':time.perf_counter()-tick})
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    report={'schema_version':'recognition-export-result/1.0','config_sha256':a.config_sha256,'paddle':paddle.__version__,
            'exports':records,'customer_samples_read':0,'training_started':False,
            'conversion_options':{'enable_optimize':False,'optimize_tool':None,'opset_version':17}}
    (target/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'exports':len(records),'paddle':paddle.__version__}))


if __name__=='__main__':main()
