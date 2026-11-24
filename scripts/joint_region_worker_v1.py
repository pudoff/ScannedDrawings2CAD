"""Blinded worker: unchanged CV/OCR/association, all predicted ROIs."""
import argparse
from pathlib import Path
import time
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, encoded, read, inside
from cad_recovery.joint_region_comparison_v1 import validate_request
from cad_recovery.preprocessing import prepare_scan
from cad_recovery.ocr_engine import LocalOCREngine
from cad_recovery.ocr import recognize_text
from cad_recovery.geometry_detection import detect_geometry
from cad_recovery.arrow_detection import detect_arrows
from cad_recovery.dimension_association import associate_dimensions

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request');parser.add_argument('--sha256');parser.add_argument('--output')
    args=parser.parse_args();request=read(checked(ROOT,{'path':args.request,'sha256':args.sha256}));validate_request(request)
    output=inside(ROOT,args.output);output.mkdir(parents=True,exist_ok=False)
    engine=LocalOCREngine(ROOT/'artifacts/models/ppocr-v5')
    class Counted:
        calls=0
        @property
        def provenance(self):return engine.provenance
        def recognize(self,image):
            self.calls+=1;return engine.recognize(image)
    counted=Counted();results=[]
    for case in request['cases']:
        saved={'id':case['id'],'crops':[]};results.append(saved)
        with Image.open(checked(ROOT,case['image'])) as original:
            assert list(original.size)==case['size_px'];original=original.convert('RGB')
            for crop in case['crops']:
                folder=output/case['id']/crop['id'];folder.mkdir(parents=True)
                source=folder/'source.png';original.crop(crop['bbox']).save(source)
                started=time.perf_counter();calls=counted.calls
                try:
                    prep=prepare_scan(source,folder,auto_deskew=False)
                    image=folder/prep['artifacts']['grayscale']
                    geometry=detect_geometry(image,folder,pixel_to_paper_mm=None,search_profile='regional_masked')
                    title=next((r['bbox_px'] for r in prep['regions'] if r['id']=='title-block-candidate'),None)
                    ocr=recognize_text(image,folder,engine=counted,pixel_to_paper_mm=None,
                        processed_to_input=prep['transforms']['processed_to_input'],title_bbox=title,
                        tile_size=1536,overlap=192,time_budget_seconds=240)
                    arrows=detect_arrows(image,folder,geometry=geometry,text_polygons=[l['polygon_px'] for l in ocr['lines']])
                    associations=associate_dimensions(image,folder,geometry=geometry,ocr=ocr,arrows=arrows)
                    row={'id':crop['id'],'error':None,'ocr':ocr,'associations':associations,
                        'geometry_count':len(geometry['candidates']),'arrow_count':len(arrows['candidates'])}
                except Exception as exc:
                    row={'id':crop['id'],'error':type(exc).__name__+': '+str(exc)}
                row.update(elapsed_seconds=time.perf_counter()-started,model_calls=counted.calls-calls)
                saved['crops'].append(row)
                result={'cases':results,'model_calls':counted.calls,'completion':False}
                (output/'results.json').write_bytes(encoded(result))
                print(case['id'],crop['id'],row['error'] or 'ok',flush=True)
    errors=sum(bool(r.get('error')) for c in results for r in c['crops'])
    complete=not errors and all(r['ocr']['coverage']['complete'] for c in results for r in c['crops'])
    (output/'results.json').write_bytes(encoded({'cases':results,'model_calls':counted.calls,'completion':complete,'errors':errors}))
    if not complete:raise SystemExit(2)

if __name__=='__main__':main()
