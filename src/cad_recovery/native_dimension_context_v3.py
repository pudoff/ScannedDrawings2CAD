"""Observe original dimension context with bounded local shaft-owner checks."""
from unittest.mock import patch

from . import native_dimension_context_v2 as previous
from .native_dimension_owner_v1 import verify


def inspect(*args, **kwargs):
    with patch.object(previous.previous, 'verify', verify):
        result = previous.inspect(*args, **kwargs)
    result['schema_version'] = 'native-dimension-context/3.0'
    result['owner_observer_version'] = 'native-dimension-owner/1.0'
    return result
