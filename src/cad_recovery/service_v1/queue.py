"""PostgreSQL leases with row locking and fencing; no in-memory queue fallback."""
from uuid import UUID,uuid4

DDL="""
CREATE SCHEMA IF NOT EXISTS cad_service_v1;
CREATE TABLE IF NOT EXISTS cad_service_v1.schema_version (version integer PRIMARY KEY CHECK(version=1));
INSERT INTO cad_service_v1.schema_version VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS cad_service_v1.storage_binding (singleton boolean PRIMARY KEY CHECK(singleton),storage_id uuid NOT NULL);
CREATE TABLE IF NOT EXISTS cad_service_v1.documents (
 id uuid PRIMARY KEY, original_name text NOT NULL, input_path text NOT NULL UNIQUE,
 input_sha256 text NOT NULL CHECK(length(input_sha256)=64), size_bytes bigint NOT NULL CHECK(size_bytes>0),
 next_version integer NOT NULL DEFAULT 2, created_at timestamptz NOT NULL DEFAULT clock_timestamp());
CREATE TABLE IF NOT EXISTS cad_service_v1.runs (
 id uuid PRIMARY KEY, job_id uuid NOT NULL REFERENCES cad_service_v1.documents(id),
 version integer NOT NULL CHECK(version>0), options jsonb NOT NULL,
 status text NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','running','completed','failed')),
 attempts integer NOT NULL DEFAULT 0, owner uuid, lease_until timestamptz,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), started_at timestamptz, finished_at timestamptz,
 manifest jsonb, summary jsonb, error jsonb, UNIQUE(job_id,version),
 CHECK((status='running')=(owner IS NOT NULL AND lease_until IS NOT NULL)),
 CHECK(status<>'completed' OR (manifest IS NOT NULL AND summary IS NOT NULL)));
CREATE INDEX IF NOT EXISTS runs_queue ON cad_service_v1.runs(status,created_at);
"""

class Queue:
    def __init__(self,settings):self.settings=settings

    def connect(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.settings.dsn,connect_timeout=5,row_factory=dict_row)

    @staticmethod
    def json(value):
        from psycopg.types.json import Jsonb
        return Jsonb(value) if value is not None else None

    def initialize(self):
        with self.connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(17441701)');c.execute(DDL)

    def ping(self):
        with self.connect() as c:return c.execute('SELECT version FROM cad_service_v1.schema_version').fetchone()['version']==1

    def bind_storage(self):
        from .storage import Storage,write_json,digest
        import json
        files=Storage(self.settings);identity=files.path('storage-id.json')
        with self.connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(17441702)')
            if not identity.exists():write_json(identity,{'storage_id':str(uuid4())})
            identifier=UUID(json.loads(identity.read_text('utf-8'))['storage_id'])
            bound=c.execute('SELECT storage_id FROM cad_service_v1.storage_binding WHERE singleton').fetchone()
            if bound:
                if bound['storage_id']!=identifier:raise ValueError('Database is bound to another artifact storage')
            else:
                # Initial adoption of an existing queue requires the actual originals.
                for doc in c.execute('SELECT input_path,input_sha256 FROM cad_service_v1.documents').fetchall():
                    if digest(files.path(doc['input_path']))!=doc['input_sha256']:raise ValueError('Original input is missing or changed')
                c.execute('INSERT INTO cad_service_v1.storage_binding VALUES(true,%s)',(identifier,))

    def create(self,record,options):
        with self.connect() as c:
            c.execute('INSERT INTO cad_service_v1.documents(id,original_name,input_path,input_sha256,size_bytes) VALUES(%s,%s,%s,%s,%s)',
                [record[k] for k in ('id','original_name','input_path','input_sha256','size_bytes')])
            return c.execute('INSERT INTO cad_service_v1.runs(id,job_id,version,options) VALUES(%s,%s,1,%s) RETURNING *',
                (uuid4(),record['id'],self.json(options))).fetchone()

    def new_version(self,job,options):
        with self.connect() as c:
            doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s FOR UPDATE',(job,)).fetchone()
            if doc is None:raise KeyError(job)
            version=doc['next_version'];c.execute('UPDATE cad_service_v1.documents SET next_version=next_version+1 WHERE id=%s',(job,))
            return c.execute('INSERT INTO cad_service_v1.runs(id,job_id,version,options) VALUES(%s,%s,%s,%s) RETURNING *',
                (uuid4(),job,version,self.json(options))).fetchone()

    def job(self,job):
        with self.connect() as c:
            doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s',(job,)).fetchone()
            if doc is None:raise KeyError(job)
            return doc,c.execute('SELECT * FROM cad_service_v1.runs WHERE job_id=%s ORDER BY version',(job,)).fetchall()

    def run(self,job,version):
        with self.connect() as c:
            run=c.execute('SELECT * FROM cad_service_v1.runs WHERE job_id=%s AND version=%s',(job,version)).fetchone()
            if run is None:raise KeyError(job)
            return run

    def claim(self):
        with self.connect() as c:
            c.execute("UPDATE cad_service_v1.runs SET status='failed',owner=NULL,lease_until=NULL,finished_at=clock_timestamp(),error=%s WHERE status='running' AND lease_until<=clock_timestamp() AND attempts>=%s",
                (self.json({'code':'attempts_exhausted','message':'Worker lease expired too many times'}),self.settings.max_attempts))
            row=c.execute("SELECT id FROM cad_service_v1.runs WHERE (status='queued' OR (status='running' AND lease_until<=clock_timestamp())) AND attempts<%s ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1",
                (self.settings.max_attempts,)).fetchone()
            if row is None:return None
            run=c.execute("UPDATE cad_service_v1.runs SET status='running',owner=%s,lease_until=clock_timestamp()+%s*interval '1 second',attempts=attempts+1,started_at=clock_timestamp() WHERE id=%s RETURNING *",
                (uuid4(),self.settings.lease_seconds,row['id'])).fetchone()
            doc=c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s',(run['job_id'],)).fetchone()
            return {**doc,**run}

    def heartbeat(self,run):
        with self.connect() as c:
            return c.execute("UPDATE cad_service_v1.runs SET lease_until=clock_timestamp()+%s*interval '1 second' WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()",
                (self.settings.lease_seconds,run['id'],run['owner'])).rowcount==1

    def finish(self,run,*,manifest=None,summary=None,error=None):
        if error is None and (not isinstance(manifest,dict) or not isinstance(summary,dict)):
            raise ValueError('Completion requires a sealed manifest and summary')
        with self.connect() as c:
            return c.execute("UPDATE cad_service_v1.runs SET status=%s,manifest=%s,summary=%s,error=%s,owner=NULL,lease_until=NULL,finished_at=clock_timestamp() WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()",
                ('failed' if error else 'completed',self.json(manifest),self.json(summary),self.json(error),run['id'],run['owner'])).rowcount==1
