"""Durable preview/apply leases and atomic, append-only corrected version publication."""
from uuid import uuid4
from .graph_correction_v1 import canonical_sha256


class CorrectionConflict(ValueError):pass


DDL="""
CREATE SCHEMA IF NOT EXISTS cad_corrections_v1;
CREATE TABLE IF NOT EXISTS cad_corrections_v1.proposals (
 id uuid PRIMARY KEY, job_id uuid NOT NULL, parent_version integer NOT NULL,
 page_index integer NOT NULL CHECK(page_index BETWEEN 0 AND 19),
 request jsonb NOT NULL, request_sha256 text NOT NULL, source_manifest_sha256 text NOT NULL,
 stage text NOT NULL CHECK(stage IN ('preview','apply')),
 status text NOT NULL CHECK(status IN ('queued','running','prepared','rejected','failed','stale','applied')),
 attempts integer NOT NULL DEFAULT 0, owner uuid, lease_until timestamptz,
 preview_manifest jsonb, preview_summary jsonb, approval jsonb, actor text,
 applied_version integer, error text, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 FOREIGN KEY(job_id,parent_version) REFERENCES cad_service_v1.runs(job_id,version),
 UNIQUE(job_id,request_sha256),
 CHECK((status='running')=(owner IS NOT NULL AND lease_until IS NOT NULL)));
"""


class CorrectionStore:
    def __init__(self,queue):self.queue=queue

    def initialize(self):
        with self.queue.connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(17441861)');c.execute(DDL)

    @staticmethod
    def _head(c,job,parent):
        doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(job,)).fetchone()
        if doc is None:raise KeyError(job)
        if doc['next_version']!=parent+1:raise CorrectionConflict('Выбрана устаревшая версия. Откройте последнюю и подготовьте правку заново.')
        return doc

    def create(self,job,parent,page,edit):
        value=edit.model_dump(mode='json');sha=canonical_sha256({'version':parent,'page':page,'edit':value})
        with self.queue.connect() as c:
            c.execute('SELECT id FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(job,))
            old=c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE job_id=%s AND request_sha256=%s',(job,sha)).fetchone()
            if old:return old
            self._head(c,job,parent)
            self._capacity(c)
            return c.execute("""INSERT INTO cad_corrections_v1.proposals
                (id,job_id,parent_version,page_index,request,request_sha256,source_manifest_sha256,stage,status)
                VALUES(%s,%s,%s,%s,%s,%s,%s,'preview','queued') RETURNING *""",
                (uuid4(),job,parent,page,self.queue.json(value),sha,edit.source_manifest_sha256)).fetchone()

    @staticmethod
    def _capacity(c):
        c.execute('SELECT pg_advisory_xact_lock(17441862)')
        if c.execute("SELECT count(*) AS n FROM cad_corrections_v1.proposals WHERE status IN ('queued','running')").fetchone()['n']>=8:
            raise CorrectionConflict('Очередь исправлений заполнена. Повторите позже.')

    def get(self,job,parent,page,identifier):
        with self.queue.connect() as c:
            row=c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE id=%s AND job_id=%s AND parent_version=%s AND page_index=%s',
                (identifier,job,parent,page)).fetchone()
            if row is None:raise KeyError(identifier)
            return row

    def list(self,job,parent,page):
        with self.queue.connect() as c:return c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE job_id=%s AND parent_version=%s AND page_index=%s ORDER BY created_at DESC LIMIT 30',
            (job,parent,page)).fetchall()

    def accept(self,row,approval,actor):
        with self.queue.connect() as c:
            c.execute('SELECT id FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(row['job_id'],))
            current=c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE id=%s FOR UPDATE',(row['id'],)).fetchone()
            value=approval.model_dump(mode='json')
            if current['approval'] is not None:
                if current['approval']!=value or current['actor']!=actor:raise CorrectionConflict('Предложение уже принято с другой записью проверки.')
                return current
            self._head(c,row['job_id'],row['parent_version'])
            if current['status']!='prepared' or current['preview_summary']['proposal_sha256']!=approval.proposal_sha256:
                raise CorrectionConflict('Нужна проверка точного готового предложения.')
            self._capacity(c)
            return c.execute("""UPDATE cad_corrections_v1.proposals SET stage='apply',status='queued',attempts=0,
                approval=%s,actor=%s,updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
                (self.queue.json(value),actor,row['id'])).fetchone()

    def claim(self):
        with self.queue.connect() as c:
            c.execute("""UPDATE cad_corrections_v1.proposals SET status='failed',owner=NULL,lease_until=NULL,
                error='attempts_exhausted' WHERE status='running' AND lease_until<=clock_timestamp() AND attempts>=3""")
            row=c.execute("""SELECT id FROM cad_corrections_v1.proposals
                WHERE (status='queued' OR (status='running' AND lease_until<=clock_timestamp())) AND attempts<3
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""").fetchone()
            if row is None:return None
            return c.execute("""UPDATE cad_corrections_v1.proposals SET status='running',owner=%s,
                lease_until=clock_timestamp()+interval '90 seconds',attempts=attempts+1,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""",(uuid4(),row['id'])).fetchone()

    def finish(self,row,*,manifest=None,summary=None,error=None):
        status='failed' if error else ('prepared' if summary['status']=='ready_for_review' else 'rejected')
        with self.queue.connect() as c:
            return c.execute("""UPDATE cad_corrections_v1.proposals SET status=%s,owner=NULL,lease_until=NULL,
                preview_manifest=COALESCE(%s,preview_manifest),preview_summary=COALESCE(%s,preview_summary),error=%s,updated_at=clock_timestamp()
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (status,self.queue.json(manifest),self.queue.json(summary),error,row['id'],row['owner'])).rowcount==1

    def publish(self,row,parent,manifest,summary):
        with self.queue.connect() as c:
            c.execute('SELECT id FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(row['job_id'],))
            current=c.execute('SELECT * FROM cad_corrections_v1.proposals WHERE id=%s FOR UPDATE',(row['id'],)).fetchone()
            if current['status']=='applied':return current['applied_version']
            if current['stage']!='apply' or current['approval'] is None:raise CorrectionConflict('Publication requires an approved apply stage')
            owned=c.execute("SELECT id FROM cad_corrections_v1.proposals WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()",
                (row['id'],row['owner'])).fetchone()
            if not owned:return None
            doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s',(row['job_id'],)).fetchone()
            if doc['next_version']!=row['parent_version']+1:
                c.execute("UPDATE cad_corrections_v1.proposals SET status='stale',owner=NULL,lease_until=NULL,error='parent_version_changed' WHERE id=%s",(row['id'],))
                return None
            live=c.execute('SELECT manifest FROM cad_service_v1.runs WHERE job_id=%s AND version=%s',(row['job_id'],row['parent_version'])).fetchone()
            if live['manifest']!=parent['manifest']:raise CorrectionConflict('Parent manifest changed')
            version=doc['next_version'];options={**parent['options'],'_manual_correction':{'id':str(row['id']),
                'parent_version':row['parent_version'],'kind':row['request']['kind'],'automatic_quality_credit':False}}
            c.execute('UPDATE cad_service_v1.documents SET next_version=next_version+1 WHERE id=%s',(row['job_id'],))
            c.execute("""INSERT INTO cad_service_v1.runs (id,job_id,version,options,status,manifest,summary,started_at,finished_at)
                VALUES(%s,%s,%s,%s,'completed',%s,%s,clock_timestamp(),clock_timestamp())""",
                (uuid4(),row['job_id'],version,self.queue.json(options),self.queue.json(manifest),self.queue.json(summary)))
            c.execute("""UPDATE cad_corrections_v1.proposals SET status='applied',owner=NULL,lease_until=NULL,
                applied_version=%s,updated_at=clock_timestamp() WHERE id=%s""",(version,row['id']))
            return version
