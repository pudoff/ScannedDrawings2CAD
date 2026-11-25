"""Existing correction schema with bounded leases and fenced publication."""
from uuid import uuid4
from datetime import datetime,timedelta,timezone
from .web_correction_store_v1 import CorrectionStore as Previous,CorrectionConflict
from .service_profiles_v12.queue import publication_timeout


class _Fenced(RuntimeError):pass


def deadline_required(deadline):
    if not isinstance(deadline,datetime) or deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError('A timezone-aware publication deadline is required')


def time_left(c,deadline):
    if not publication_timeout(c,deadline):raise _Fenced()
    if not c.execute('SELECT clock_timestamp()<%s AS valid',(deadline,)).fetchone()['valid']:raise _Fenced()


class CorrectionStore(Previous):
    def claim(self):
        with self.queue.connect_worker() as c:
            c.execute("""UPDATE cad_corrections_v1.proposals SET status='failed',owner=NULL,lease_until=NULL,
                error='attempts_exhausted' WHERE status='running' AND lease_until<=clock_timestamp() AND attempts>=3""")
            row=c.execute("""SELECT id FROM cad_corrections_v1.proposals WHERE
                (status='queued' OR (status='running' AND lease_until<=clock_timestamp())) AND attempts<3
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""").fetchone()
            if row is None:return None
            return c.execute("""UPDATE cad_corrections_v1.proposals SET status='running',owner=%s,
                lease_until=clock_timestamp()+%s*interval '1 second',attempts=attempts+1,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
                (uuid4(),self.queue.settings.lease_seconds,row['id'])).fetchone()

    def heartbeat(self,row):
        with self.queue.lease_connection(row,'correction') as c:
            publication_timeout(c,datetime.now(timezone.utc)+timedelta(seconds=2))
            return c.execute("""UPDATE cad_corrections_v1.proposals
                SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (self.queue.settings.lease_seconds,row['id'],row['owner'])).rowcount==1

    def close_heartbeat(self,row):self.queue.close_heartbeat(row,'correction')

    def finish(self,row,*,manifest=None,summary=None,error=None,deadline=None):
        if error is not None:
            with self.queue.lease_connection(row,'correction') as c:
                publication_timeout(c,datetime.now(timezone.utc)+timedelta(seconds=2))
                return c.execute("""UPDATE cad_corrections_v1.proposals
                    SET status='failed',owner=NULL,lease_until=NULL,error=%s,updated_at=clock_timestamp()
                    WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                    (error,row['id'],row['owner'])).rowcount==1
        deadline_required(deadline)
        if row['stage']!='preview' or not isinstance(manifest,dict) or not isinstance(summary,dict):
            raise ValueError('Preview manifest and summary required')
        status='prepared' if summary['status']=='ready_for_review' else 'rejected'
        if deadline<=datetime.now(timezone.utc):return False
        try:
            with self.queue.lease_connection(row,'correction') as c:
                time_left(c,deadline)
                doc=c.execute('SELECT next_version FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(row['job_id'],)).fetchone()
                if doc is None or doc['next_version']!=row['parent_version']+1:raise _Fenced()
                live=c.execute('SELECT manifest FROM cad_service_v1.runs WHERE job_id=%s AND version=%s',
                    (row['job_id'],row['parent_version'])).fetchone()
                if not live or live['manifest']['sha256']!=row['source_manifest_sha256']:raise _Fenced()
                time_left(c,deadline)
                n=c.execute("""UPDATE cad_corrections_v1.proposals SET status=%s,owner=NULL,lease_until=NULL,
                    preview_manifest=%s,preview_summary=%s,error=NULL,updated_at=clock_timestamp()
                    WHERE id=%s AND owner=%s AND stage='preview' AND status='running' AND lease_until>clock_timestamp()
                    AND request_sha256=%s AND clock_timestamp()<%s""",
                    (status,self.queue.json(manifest),self.queue.json(summary),row['id'],row['owner'],row['request_sha256'],deadline)).rowcount
                if n!=1:raise _Fenced()
                time_left(c,deadline)
            return True
        except _Fenced:return False

    def publish(self,row,parent,manifest,summary,*,deadline=None):
        deadline_required(deadline)
        if row['stage']!='apply' or row['approval'] is None:
            raise CorrectionConflict('Publication requires approved apply stage')
        if deadline<=datetime.now(timezone.utc):return None
        try:
            with self.queue.lease_connection(row,'correction') as c:
                time_left(c,deadline)
                doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(row['job_id'],)).fetchone()
                current=c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE id=%s FOR UPDATE',(row['id'],)).fetchone()
                if current is None:raise _Fenced()
                if current['status']=='applied':return current['applied_version']
                owned=c.execute("""SELECT id FROM cad_corrections_v1.proposals WHERE id=%s AND owner=%s
                    AND status='running' AND stage='apply' AND lease_until>clock_timestamp()
                    AND request_sha256=%s AND clock_timestamp()<%s""",
                    (row['id'],row['owner'],row['request_sha256'],deadline)).fetchone()
                if not owned or doc is None:raise _Fenced()
                if current['approval']!=row['approval']:raise CorrectionConflict('Approval changed')
                if doc['next_version']!=row['parent_version']+1:
                    time_left(c,deadline)
                    c.execute("""UPDATE cad_corrections_v1.proposals SET status='stale',owner=NULL,lease_until=NULL,
                        error='parent_version_changed' WHERE id=%s AND owner=%s""",(row['id'],row['owner']))
                    time_left(c,deadline)
                    return None
                live=c.execute('SELECT manifest FROM cad_service_v1.runs WHERE job_id=%s AND version=%s',
                    (row['job_id'],row['parent_version'])).fetchone()
                if not live or live['manifest']!=parent['manifest'] or parent['manifest']['sha256']!=row['source_manifest_sha256']:
                    raise CorrectionConflict('Parent manifest changed')
                version=doc['next_version']
                options={**parent['options'],'_manual_correction':dict(id=str(row['id']),parent_version=row['parent_version'],
                    kind=row['request']['kind'],automatic_quality_credit=False)}
                time_left(c,deadline)
                n=c.execute("""UPDATE cad_service_v1.documents SET next_version=next_version+1
                    WHERE id=%s AND next_version=%s AND clock_timestamp()<%s""",(row['job_id'],version,deadline)).rowcount
                if n!=1:raise _Fenced()
                n=c.execute("""INSERT INTO cad_service_v1.runs
                    (id,job_id,version,options,status,manifest,summary,started_at,finished_at)
                    SELECT %s,%s,%s,%s,'completed',%s,%s,clock_timestamp(),clock_timestamp()
                    WHERE clock_timestamp()<%s""",(uuid4(),row['job_id'],version,self.queue.json(options),self.queue.json(manifest),self.queue.json(summary),deadline)).rowcount
                if n!=1:raise _Fenced()
                n=c.execute("""UPDATE cad_corrections_v1.proposals SET status='applied',owner=NULL,lease_until=NULL,
                    applied_version=%s,updated_at=clock_timestamp() WHERE id=%s AND owner=%s
                    AND status='running' AND lease_until>clock_timestamp() AND clock_timestamp()<%s""",
                    (version,row['id'],row['owner'],deadline)).rowcount
                if n!=1:raise _Fenced()
                time_left(c,deadline)
            return version
        except _Fenced:return None
