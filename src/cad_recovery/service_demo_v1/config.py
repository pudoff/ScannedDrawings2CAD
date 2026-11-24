"""Explicit presentation options; historical service contracts stay immutable."""
import os
from pathlib import Path
from typing import Literal

from cad_recovery.service_v1.config import ProcessingOptions as HistoricalOptions
from cad_recovery.service_v1.config import Settings as HistoricalSettings


class ProcessingOptions(HistoricalOptions):
    profile: Literal['presentation_draft', 'baseline', 'geometry_observations'] = 'presentation_draft'


class Settings(HistoricalSettings):
    @classmethod
    def environment(cls):
        return cls(Path(os.environ.get('CAD_SERVICE_STORAGE', 'artifacts/demo-service-v1/storage')),
                   os.environ.get('CAD_SERVICE_DSN', ''),
                   timeout_seconds=float(os.environ.get('CAD_SERVICE_TIMEOUT_SECONDS', '900')),
                   token=os.environ.get('CAD_SERVICE_TOKEN'))
