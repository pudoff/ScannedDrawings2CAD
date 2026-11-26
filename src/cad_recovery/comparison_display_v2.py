"""Display-only SVG correction; downloaded CAD and coordinates remain untouched."""
from pathlib import Path
from types import SimpleNamespace
import json,subprocess,sys,tempfile
from .native_comparison_v1 import NativeComparisons,native_frame
from .comparison_service_v1 import ROOT,AttemptBudget
from .web_correction_contract_v1 import decode


class DisplayComparisons(NativeComparisons):
    def _render(self,path,digest,render_box):
        report=decode((path.parent/'report.json').read_bytes())
        exported=report['pages'][int(path.stem.split('-')[-1])-1].get('demo_export',{})
        if exported.get('mode')=='native_annotations_draft':render_box,_=native_frame(exported)
        if exported.get('mode')=='reviewed_geometry':render_box=None
        with tempfile.TemporaryDirectory(prefix='display-v2-',dir=self.work) as directory:
            folder=Path(directory)
            if not folder.resolve().is_relative_to(self.work.resolve()):raise ValueError('Invalid derived directory')
            request=folder/'request.json';output=folder/'render.json'
            request.write_text(json.dumps(dict(path=str(path),sha256=digest,render_box=render_box)),encoding='utf-8')
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/render_comparison_v2.py'),
                '--request',str(request),'--output',str(output)],cwd=ROOT,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            budget=None
            try:
                budget=AttemptBudget(child,folder,self.files,SimpleNamespace(timeout_seconds=20,
                    max_output_bytes=32*1024**2,max_output_files=4),memory_MiB=512)
                with self.lock:
                    if self.stopped:child.terminate()
                    self.children.add(child)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait();return dict(status='unavailable',reason='render_timeout')
                if code or not output.is_file():return dict(status='unavailable',reason='unsupported_or_limited_dxf')
                budget.check()
                if output.stat().st_size>32*1024**2:return dict(status='unavailable',reason='render_output_limit')
                value=json.loads(output.read_text(encoding='utf-8'))
                if value['dxf_sha256']!=digest:raise ValueError('Render context mismatch')
                value['render_budget']=budget.measurements()
                value['status']='ready' if 'svg' in value else 'counts_only'
                return value
            finally:
                if budget is not None:budget.stop()
                if child.poll() is None:child.kill();child.wait()
                with self.lock:self.children.discard(child)


def inspector_javascript(source):
    old="opacity:Number($('layer-opacity').value)/100,class:'comparison-dxf'"
    new="opacity:$('layer-source').checked?Number($('layer-opacity').value)/100:1,style:$('layer-source').checked?'':'filter:none',class:'comparison-dxf'"
    if source.count(old)!=1:raise ValueError('DXF display hook changed')
    source=source.replace(old,new)
    hook="    const maskInfo=comparison?.masks[$('comparison-mask').value];"
    if source.count(hook)!=1:raise ValueError('DXF opacity control hook changed')
    display_controls="""    $('layer-opacity').disabled=!$('layer-source').checked;
    const dxfLegend=canvas.closest('.inspection-visual').querySelector('.fine-print');
    if(dxfLegend)dxfLegend.textContent=$('layer-source').checked
      ? 'DXF — синий поверх исходника; наблюдения — цветные, требуют проверки.'
      : 'DXF — чёрный; наблюдения — цветные, требуют проверки.';
"""
    return source.replace(hook,display_controls+hook)
