import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_grid_role_v2 import save_owner


def test_actual_owner_png_round_trip_preserves_zero_and_high_source_ids(tmp_path):
    owner=np.array([[0,1,4096],[2048,33,255]],np.int32);path=tmp_path/'owners.png'
    assert save_owner(path,owner,4096) and np.array_equal(np.array(Image.open(path)),owner)


@pytest.mark.parametrize('bad',[np.array([[-1]],np.int32),np.array([[4097]],np.int32)])
def test_out_of_domain_owner_ids_are_rejected_before_png_write(tmp_path,bad):
    with pytest.raises(ValueError):save_owner(tmp_path/'bad.png',bad,4096)
