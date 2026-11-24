import os
from cad_recovery.service_v1.queue import Queue as HistoricalQueue
from cad_recovery.profile_contract_v2 import ProcessingOptions,resolve_profile,verify_models

class Queue(HistoricalQueue):
    @staticmethod
    def bound_options(options):
        parsed=ProcessingOptions.model_validate(options);binding=resolve_profile(parsed.profile)
        verify_models(binding,os.environ.get('CAD_DEMO_OCR_MODELS'))
        return {**parsed.model_dump(),'_profile_binding':binding}

    def create(self,record,options):return super().create(record,self.bound_options(options))
    def new_version(self,job,options):return super().new_version(job,self.bound_options(options))
