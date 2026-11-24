from dataclasses import dataclass,field
from pathlib import Path
import os
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field

class ProcessingOptions(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    profile:Literal['baseline','geometry_observations']='baseline'
    render_dpi:float=Field(default=120.,ge=36,le=300)
    image_dpi:float|None=Field(default=None,ge=36,le=1200)
    rotation_deg:Literal[0,90,180,270]=0

@dataclass(frozen=True)
class Settings:
    root:Path
    dsn:str=field(repr=False)
    max_upload_bytes:int=50*1024*1024
    max_output_bytes:int=250*1024*1024
    max_output_files:int=2000
    lease_seconds:int=30
    timeout_seconds:float=180.
    max_attempts:int=3
    token:str|None=field(default=None,repr=False)

    def __post_init__(self):
        object.__setattr__(self,'root',self.root.resolve())
        if not self.dsn or min(self.max_upload_bytes,self.max_output_bytes,self.max_output_files,self.lease_seconds,self.timeout_seconds,self.max_attempts)<=0:
            raise ValueError('Positive service limits and a PostgreSQL DSN are required')

    @classmethod
    def environment(cls):
        return cls(Path(os.environ.get('CAD_SERVICE_STORAGE','artifacts/service-data-v1')),
            os.environ.get('CAD_SERVICE_DSN',''),token=os.environ.get('CAD_SERVICE_TOKEN'))
