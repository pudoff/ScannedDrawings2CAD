"""Keep bounded SQL/publication semantics; bind new layer-profile submissions."""
import os
from cad_recovery.service_profiles_v12.queue import Queue as Previous
from cad_recovery.profile_contract_v7 import ProcessingOptions, resolve_profile, verify_models


class Queue(Previous):
    @staticmethod
    def bound_options(options):
        parsed = ProcessingOptions.model_validate(options)
        binding = resolve_profile(parsed.profile)
        verify_models(binding, os.environ.get('CAD_DEMO_OCR_MODELS'))
        return {**parsed.model_dump(), '_profile_binding': binding}
