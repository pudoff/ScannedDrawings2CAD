import os
import time
from cad_recovery.service_profiles_v2.queue import Queue as PreviousQueue
from cad_recovery.profile_contract_v4 import ProcessingOptions, resolve_profile, verify_models


class Queue(PreviousQueue):
    def connect(self):
        """Retry only establishment, before any transaction/SQL has run."""
        import psycopg
        from psycopg.rows import dict_row
        for attempt in range(2):
            try:
                return psycopg.connect(self.settings.dsn,connect_timeout=15,row_factory=dict_row)
            except (psycopg.OperationalError,psycopg.InterfaceError):
                if attempt:
                    raise
                time.sleep(.1)

    @staticmethod
    def bound_options(options):
        parsed = ProcessingOptions.model_validate(options)
        binding = resolve_profile(parsed.profile)
        verify_models(binding, os.environ.get('CAD_DEMO_OCR_MODELS'))
        return {**parsed.model_dump(), '_profile_binding':binding}
