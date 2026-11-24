"""Bounded retry of transient initial heartbeat errors; ownership stays fenced.

A false heartbeat is never retried. No child starts without a successful SQL
renewal. Subsequent watchdog renewals keep the unchanged V1 fail-closed policy.
"""
import time

import psycopg

from .lease_keeper_v1 import LeaseKeeper as Previous
from .runtime_worker_v1 import LeaseLost


class LeaseKeeper(Previous):
    def __init__(self, heartbeat, seconds, *, report=lambda value: None, clock=time.monotonic, sleep=time.sleep):
        super().__init__(heartbeat, seconds)
        self.report = report; self.clock = clock; self.sleep = sleep
        self.initial_events = []

    def renew(self):
        if self.thread is not None:
            return super().renew()
        began = self.clock()
        # Includes connection-call cost; max three calls, at most five seconds
        # to start another call, and never beyond one third of the lease.
        window = min(5., self.seconds/3)
        for attempt in range(1, 4):
            if attempt > 1 and self.clock()-began >= window:
                self.lost = True
                self.initial_events[-1].update(retry=False, retry_window_exhausted=True)
                self.report({'schema_version': 'initial-lease/2.0', 'attempts': self.initial_events,
                    'initial_ownership_verified': False, 'maximum_calls': 3,
                    'retry_start_window_seconds': window, 'false_result_retried': False})
                raise LeaseLost()
            started = self.clock(); valid = False; transient = False; kind = None
            try:
                valid = self.heartbeat()
            except (psycopg.OperationalError, psycopg.InterfaceError) as error:
                transient = True; kind = type(error).__name__
            except Exception as error:
                kind = type(error).__name__
            retry = transient and attempt < 3 and self.clock()-began < window
            self.initial_events.append({'attempt': attempt, 'verified': bool(valid),
                'transient_error': transient, 'error_type': kind, 'retry': retry,
                'elapsed_seconds': self.clock()-began})
            self.report({'schema_version': 'initial-lease/2.0', 'attempts': self.initial_events,
                'initial_ownership_verified': bool(valid), 'maximum_calls': 3,
                'retry_start_window_seconds': window, 'false_result_retried': False})
            if valid:
                self.last = started
                return
            if not retry:
                self.lost = True
                raise LeaseLost()
            self.sleep(min(.05, max(0., window-(self.clock()-began))))
        raise LeaseLost()
