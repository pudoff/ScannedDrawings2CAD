import os
from cad_recovery.service_profiles_v10.queue import Queue as PriorQueue
from cad_recovery.profile_contract_v6 import ProcessingOptions,resolve_profile,verify_models


class Queue(PriorQueue):
    @staticmethod
    def bound_options(options):
        parsed=ProcessingOptions.model_validate(options);binding=resolve_profile(parsed.profile)
        verify_models(binding,os.environ.get('CAD_DEMO_OCR_MODELS'))
        return {**parsed.model_dump(),'_profile_binding':binding}
