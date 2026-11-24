"""Persist the source chain and contact ledger alongside unchanged role rules."""
from pathlib import Path
from unittest.mock import patch

from . import dimension_role_stage_v5 as previous
from .dimension_role_stage_v1 import write
from .joint_dimension_roles_v5 import propose


def analyze(folder, *args, **kwargs):
    observed = {}

    def with_chain(image, **options):
        result = propose(image, **options)
        observed['chain'] = result['native_chain']
        return result

    with patch.object(previous,'propose',with_chain):
        result = previous.analyze(folder,*args,**kwargs)
    write(Path(folder)/'native-dimension-chain.json',observed['chain'])
    return result
