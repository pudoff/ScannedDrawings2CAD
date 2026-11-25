"""Retain V9 sealing/scans; make every upload I/O stage cancellation safe."""
from cad_recovery.service_profiles_v9.storage import Storage as Previous
from cad_recovery.upload_io_v1 import upload


class Storage(Previous):
    async def upload(self,name,chunks):
        return await upload(self,name,chunks)
