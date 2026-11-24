import os
from pathlib import Path
from cad_recovery.service_demo_v1.config import Settings as HistoricalSettings

class Settings(HistoricalSettings):
    @classmethod
    def environment(cls):
        return cls(Path(os.environ.get('CAD_SERVICE_STORAGE','artifacts/profile-service-v1/storage')),
            os.environ.get('CAD_SERVICE_DSN',''),timeout_seconds=float(os.environ.get('CAD_SERVICE_TIMEOUT_SECONDS','900')),
            max_output_bytes=512*1024**2,max_output_files=8000,token=os.environ.get('CAD_SERVICE_TOKEN'))
