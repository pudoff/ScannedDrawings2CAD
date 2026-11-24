"""Configurable leases; transactionally fence table parent and deadline."""
from uuid import uuid4
from datetime import datetime,timedelta,timezone
from .web_tables_store_v1 import TableStore as Previous
from .service_profiles_v12.queue import publication_timeout


class TableStore(Previous):
    def claim(self):
        with self.queue.connect() as c:
            c.execute("""UPDATE cad_tables_v1.proposals SET status='failed',owner=NULL,lease_until=NULL,error='attempts_exhausted'
                WHERE status='running' AND lease_until<=clock_timestamp() AND attempts>=3""")
            row=c.execute("""SELECT id FROM cad_tables_v1.proposals WHERE
                (status='queued' OR (status='running' AND lease_until<=clock_timestamp())) AND attempts<3
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""").fetchone()
            if row is None:return None
            return c.execute("""UPDATE cad_tables_v1.proposals SET status='running',owner=%s,
                lease_until=clock_timestamp()+%s*interval '1 second',attempts=attempts+1,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
                (uuid4(),self.queue.settings.lease_seconds,row['id'])).fetchone()

    def heartbeat(self,row):
        with self.queue.lease_connection(row,'table') as c:
            publication_timeout(c,datetime.now(timezone.utc)+timedelta(seconds=2))
            return c.execute("""UPDATE cad_tables_v1.proposals SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (self.queue.settings.lease_seconds,row['id'],row['owner'])).rowcount==1

    def finish(self,row,*,manifest=None,summary=None,error=None,deadline=None):
        if error is not None:return super().finish(row,error=error)
        if not isinstance(manifest,dict) or not isinstance(summary,dict) or deadline is None:
            raise ValueError('Completion requires manifest, summary and deadline')
        with self.queue.lease_connection(row,'table') as c:
            if not publication_timeout(c,deadline):return False
            # Same lock order as publish/new_version: document, then proposal.
            doc=c.execute('SELECT next_version FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(row['job_id'],)).fetchone()
            if doc is None or doc['next_version']!=row['parent_version']+1:return False
            if not publication_timeout(c,deadline):return False
            return c.execute("""UPDATE cad_tables_v1.proposals SET status=%s,owner=NULL,lease_until=NULL,
                preview_manifest=%s,summary=%s,error=NULL,updated_at=clock_timestamp()
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()
                AND clock_timestamp()<%s""",
                ('prepared' if summary['has_changes'] else 'rejected',self.queue.json(manifest),
                 self.queue.json(summary),row['id'],row['owner'],deadline)).rowcount==1

    def close_heartbeat(self,row):self.queue.close_heartbeat(row,'table')
