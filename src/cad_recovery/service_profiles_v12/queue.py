"""Bound worker SQL waits and fence result publication by attempt deadline."""
from datetime import datetime, timedelta, timezone
import time
import threading
from contextlib import contextmanager
from cad_recovery.service_profiles_v11.queue import Queue as Previous


def publication_timeout(connection, deadline):
    remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return False
    milliseconds = max(1, min(2000, int(remaining*1000)))
    connection.execute("SELECT set_config('statement_timeout',%s,true)", (str(milliseconds),))
    return True


class Queue(Previous):
    def __init__(self,settings):
        super().__init__(settings)
        self._leases={};self._lease_lock=threading.Lock()

    def _connect(self,seconds):
        import psycopg
        from psycopg.rows import dict_row
        # No retry of a transaction or ambiguous COMMIT. These settings also
        # bound heartbeat/row-lock waits while the independent watchdog runs.
        for attempt in range(2):
            try:
                return psycopg.connect(self.settings.dsn, connect_timeout=seconds, row_factory=dict_row,
                    options='-c statement_timeout=10000 -c lock_timeout=1000 -c idle_in_transaction_session_timeout=15000')
            except (psycopg.OperationalError,psycopg.InterfaceError):
                if attempt:raise
                time.sleep(.05)

    def connect(self):return self._connect(15)

    def connect_worker(self):return self._connect(2)

    @contextmanager
    def lease_connection(self,run,kind='profile'):
        key=(kind,str(run['id']),str(run['owner']))
        # One connection per owned attempt, serial use by keeper and final
        # publication. Avoid repeated Windows PostgreSQL backend startup.
        with self._lease_lock:
            c=self._leases.get(key)
            if c is None or c.closed:
                c=self.connect_worker();self._leases[key]=c
            try:
                with c.transaction():yield c
            except BaseException:
                self._leases.pop(key,None);c.close();raise

    def close_heartbeat(self,run,kind='profile'):
        key=(kind,str(run['id']),str(run['owner']))
        with self._lease_lock:
            c=self._leases.pop(key,None)
            if c is not None:c.close()

    def heartbeat(self,run):
        with self.lease_connection(run) as c:
            publication_timeout(c,datetime.now(timezone.utc)+timedelta(seconds=2))
            return c.execute("""UPDATE cad_service_v1.runs
                SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (self.settings.lease_seconds,run['id'],run['owner'])).rowcount==1

    def finish(self, run, *, manifest=None, summary=None, error=None, deadline=None):
        if error is not None:
            return super().finish(run, error=error)
        if not isinstance(manifest,dict) or not isinstance(summary,dict) or deadline is None:
            raise ValueError('Completion requires manifest, summary and deadline')
        with self.lease_connection(run) as c:
            if not publication_timeout(c, deadline):
                return False
            return c.execute("""UPDATE cad_service_v1.runs SET status='completed',manifest=%s,summary=%s,
                error=NULL,owner=NULL,lease_until=NULL,finished_at=clock_timestamp()
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()
                AND clock_timestamp()<%s""",
                (self.json(manifest),self.json(summary),run['id'],run['owner'],deadline)).rowcount==1
