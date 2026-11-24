"""Reuse bounded V24 attempt lifecycle and publication with v8 validation."""
from pathlib import Path
from unittest.mock import patch
from cad_recovery.service_profiles_v17 import worker as previous
from cad_recovery.service_profiles_v21.queue import Queue
from cad_recovery.profile_contract_v9 import validate_binding


def process_once(settings,queue=None,storage=None,*,runner=None):
    # Dedicated serial worker; the child dispatches the original stored binding.
    with patch.object(previous,'validate_binding',validate_binding):
        return previous.process_once(settings,queue=queue or Queue(settings),storage=storage,
            runner=runner or Path(__file__).resolve().parents[3]/'scripts/service_profile_diagnostic_v7.py')
