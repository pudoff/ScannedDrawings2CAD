"""Keep admission safeguards while rechecking local source shaft ownership."""
from unittest.mock import patch

from . import native_head_admission_v1 as previous
from .native_dimension_context_v3 import inspect


def build(*args, **kwargs):
    with patch.object(previous, 'inspect', inspect):
        result = previous.build(*args, **kwargs)
    result['report']['tip_admission']['schema_version'] = 'native-head-admission/2.0'
    return result
