"""Heartbeat independent of process startup and artifact I/O in the worker."""
import threading
import time
from .runtime_worker_v1 import LeaseLost


class LeaseKeeper:
    def __init__(self,heartbeat,seconds):
        self.heartbeat=heartbeat;self.seconds=seconds;self.last=None
        self.lost=False;self.done=threading.Event();self.thread=None

    def renew(self):
        started=time.monotonic()
        try:valid=self.heartbeat()
        except Exception:valid=False
        if not valid:
            self.lost=True
            raise LeaseLost()
        self.last=started

    def start(self):
        self.renew()
        self.thread=threading.Thread(target=self._watch,name='attempt-lease',daemon=True)
        self.thread.start()

    def _watch(self):
        while not self.done.wait(min(2.,self.seconds/3)):
            try:self.renew()
            except LeaseLost:return

    def alive(self):
        return not self.lost and self.last is not None and time.monotonic()-self.last<self.seconds

    def stop(self,*,strict=True):
        self.done.set()
        if self.thread is not None:self.thread.join(timeout=5)
        if strict and ((self.thread is not None and self.thread.is_alive()) or not self.alive()):
            raise LeaseLost()
