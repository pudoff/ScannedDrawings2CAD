"""Freeze all 69 field slots and inherited uncertainty before line-crop review."""
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy

from cad_recovery.clear_pairs import digest
from cad_recovery.label_release_v1 import checked_pin

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'configs/reviews/ocr-control-lines-protocol-v2.json'
PARENT={'path':'artifacts/ocr-control-scope-final-v3.json','sha256':'4d15f21f43d6b3828e6e0915419e68bf8219537b21b5db1a0fd26056afffd72a'}
SCOPE={'path':'configs/reviews/ocr-control-protocol-v3.json','sha256':'3b6d60097cb888af0d2c0625a5f6803ced3d7e86c05a527a2d6478effa253270'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}


def main():
    assert not OUTPUT.exists();checked_pin(ROOT,PARENT)
    scope_runner=runpy.run_path(str(ROOT/'scripts/build_ocr_control_scope_v3.py'),run_name='line_scope_prerequisite')
    assert scope_runner['verify_protocol'](ROOT,SCOPE)['reserved_source_count']==23
    scope=read(checked_pin(ROOT,SCOPE));probes=read(checked_pin(ROOT,scope['review_probes']))
    sources_by_id={r['id']:r for r in probes['sources']};sources=[]
    for reservation in scope['reservations']:
        for identifier in reservation['source_ids']:
            row=sources_by_id[identifier]
            sources.append({**{k:row[k] for k in ('index','id','input_path','input_sha256','group_id','preparation','source_size_px')},
                'reserved_split':reservation['reserved_split'],
                'parents':{r['name']:{'box_source_xyxy':r['box_source_px'],'image':r['image']} for r in row['regions']},
                'inherited_uncertain_fields':['designation'] if reservation['reserved_split']=='validation' else ['adjacent_reference']})
    assert len(sources)==23
    result={'schema_version':'ocr-control-lines-protocol/2.0','task':'P14.1b.7c','parent_final':PARENT,
        'scope_protocol':SCOPE,'scope_manifest':scope['scope_manifest'],'source_probes':scope['review_probes'],
        'source_readings':scope['review_readings'],
        'previous_ocr_manifest':{'path':'artifacts/ocr-release-v3/manifest.json','sha256':'40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'},
        'created_before_line_review_utc':datetime.now(timezone.utc).isoformat(),
        'extractor':pin(Path(__file__)),'fields':['designation','part_name','adjacent_reference'],
        'candidate_count':69,'sources':sources,
        'inherited_uncertainty_policy':'Known uncertain fields retain text=null; do not silently clear by filename, sibling, font convention or model output.',
        'absent_field_policy':'Keep an explicit absent record linked to the reviewed parent; no crop or OCR target is invented.',
        'crop_policy':'Exact original RGB rectangle, positive white margins and no line carrier for accepted text; quarter-turn rotations only.',
        'allowed_rotations_cw':[0,90,180,270],'glyph_policy':'Literal visible text only; no restoration from metadata.',
        'complete_registry':'All prior 90 train records and all whole-group train/control commitments accompany the release.',
        'OCR_run':False,'training_started':False,'human_verified':False,'native_CAD_opened':False,
        'independent_evaluation':False,'physical_geometry_approved':False}
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pin(OUTPUT)))


if __name__=='__main__':main()
