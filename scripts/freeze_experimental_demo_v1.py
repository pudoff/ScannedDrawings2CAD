"""Pin the new demo adapter and the already frozen experiment code, without retraining."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import pin, checked
from cad_recovery.profile_contract_v3 import DEFINITION, REGISTRY

NEW = [
    'src/cad_recovery/experimental_cv_v1.py', 'src/cad_recovery/experimental_export_v1.py',
    'src/cad_recovery/profile_contract_v3.py', 'src/cad_recovery/profile_pipeline_v3.py',
    'src/cad_recovery/service_profiles_v8/__init__.py', 'src/cad_recovery/service_profiles_v8/queue.py',
    'src/cad_recovery/service_profiles_v8/api.py', 'src/cad_recovery/service_profiles_v8/worker.py',
    'scripts/service_profile_pipeline_v3.py', 'scripts/run_profiles_v10.py',
    'scripts/freeze_experimental_demo_v1.py',
    'artifacts/cv-skeleton-runtime-v1/installed-files.json',
    'src/cad_recovery/annotation_style_v1.py',
    'src/cad_recovery/service_profiles_v6/api.py',
    'src/cad_recovery/service_profiles_v7/assets/index.html',
    'src/cad_recovery/service_profiles_v7/assets/app.js',
    'src/cad_recovery/service_profiles_v7/assets/inspector.js',
    'src/cad_recovery/service_profiles_v7/assets/corrections.js',
    'src/cad_recovery/service_profiles_v7/assets/workspace.css',
]


def main():
    experiment={'path':'artifacts/native-frame-profile-v3/plan.json',
        'sha256':'2e829327f4f0f55ac55d234fc14a5214c0fccaec4ba95e9484b625c72be98523'}
    plan=json.loads(checked(ROOT,experiment).read_text('utf-8'))
    code={p['path']:p['sha256'] for p in plan['code']}
    for path,sha in code.items():
        checked(ROOT,{'path':path,'sha256':sha})
    # Transitive source producers predate the compact current experiment code list.
    for name in ('source_chain_inputs_v1','boundary_role_inputs_v1','contact_role_inputs_v1',
                 'run_stroke_ownership_v1','run_supported_role_overlay_v1','run_topology_robust_v1',
                 'run_head_recall_v1'):
        NEW.append('scripts/'+name+'.py')
    for path in NEW:
        code[path]=pin(ROOT,path)['sha256']
    REGISTRY.write_text(json.dumps({'schema_version':'processing-profiles/3.0','definition':DEFINITION,
        'experiment':experiment,'code':code},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(pin(ROOT,REGISTRY.relative_to(ROOT).as_posix()))


if __name__=='__main__':main()
