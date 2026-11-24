import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_case_batches_v1 import validate_rows


def fixture():
    return {'members': [{'case': {'id': i}, 'views': [{'mode': m} for m in ('native_crop', 'half_resolution_crop')]}
                        for i in ('a', 'b')]}


def test_all_case_view_pairs_are_required_independently_of_order():
    rows = [{'case': i, 'mode': m} for i in ('a', 'b') for m in ('native_crop', 'half_resolution_crop')]
    validate_rows(fixture(), rows[::-1])


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unexpected'])
def test_incomplete_or_misassigned_batch_is_rejected(kind):
    rows = [{'case': i, 'mode': m} for i in ('a', 'b') for m in ('native_crop', 'half_resolution_crop')]
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows[-1] = rows[0]
    if kind == 'unexpected': rows[-1] = {'case': 'outside', 'mode': 'native_crop'}
    with pytest.raises(ValueError): validate_rows(fixture(), rows)
