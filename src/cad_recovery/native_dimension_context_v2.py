"""Original context with bounded alternate ARC flanks; semantic rules unchanged."""
from unittest.mock import patch

from . import native_dimension_context_v1 as previous
from .native_dimension_arc_v2 import observe


def inspect(*args, **kwargs):
    with patch.object(previous, 'observe_arcs', observe):
        result = previous.inspect(*args, **kwargs)
    result['schema_version'] = 'native-dimension-context/2.0'
    result['arc_observer_version'] = 'native-dimension-arc/2.0'
    return result
