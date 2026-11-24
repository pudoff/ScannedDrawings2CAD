import ast
from pathlib import Path
from cad_recovery import local_role_guard_v3

ROOT=Path(__file__).resolve().parents[1]


def functions(path):
    return [ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]


def test_bundle_branch_has_its_declared_opencv_dependency():
    assert callable(local_role_guard_v3.cv2.pointPolygonTest)
    # Exercise the exact previously missing symbol on a polygon, not just import.
    assert local_role_guard_v3.cv2.pointPolygonTest(local_role_guard_v3.np.array([[0,0],[8,0],[8,8],[0,8]],'float32'),(4.,4.),True)>0


def test_import_repair_preserves_function_bodies_and_parameters():
    for old,new in [('local_role_guard_v2.py','local_role_guard_v3.py'),('source_chain_profile_v1.py','source_chain_profile_v2.py')]:
        assert functions(ROOT/'src/cad_recovery'/old)==functions(ROOT/'src/cad_recovery'/new)
    from cad_recovery.local_role_guard_v2 import POLICY as original
    assert original==local_role_guard_v3.POLICY
