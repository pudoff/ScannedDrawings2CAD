"""Durable table-review queue; leases fence preview publication and cancellation."""
from uuid import uuid4
import hashlib
from .dataset_layout_v1 import encoded


class TableConflict(ValueError): pass


DDL = """
CREATE SCHEMA IF NOT EXISTS cad_tables_v1;
CREATE TABLE IF NOT EXISTS cad_tables_v1.proposals (
 id uuid PRIMARY KEY, job_id uuid NOT NULL, parent_version integer NOT NULL,
 page_index integer NOT NULL CHECK(page_index BETWEEN 0 AND 19),
 request jsonb NOT NULL, request_sha256 text NOT NULL,
 status text NOT NULL CHECK(status IN ('queued','running','prepared','rejected','failed','cancelled','applied')),
 attempts integer NOT NULL DEFAULT 0, owner uuid, lease_until timestamptz,
 preview_manifest jsonb, summary jsonb, approval jsonb, applied_version integer, error text,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 FOREIGN KEY(job_id,parent_version) REFERENCES cad_service_v1.runs(job_id,version),
 UNIQUE(job_id,request_sha256),
 CHECK((status='running')=(owner IS NOT NULL AND lease_until IS NOT NULL)));
"""


class TableStore:
    def __init__(self, queue): self.queue = queue

    def initialize(self):
        with self.queue.connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(17441901)'); c.execute(DDL)

    def _head(self, c, job, parent):
        row = c.execute('SELECT * FROM cad_service_v1.documents WHERE id=%s FOR UPDATE', (job,)).fetchone()
        if row is None: raise KeyError(job)
        if row['next_version'] != parent+1: raise TableConflict('Откройте последнюю версию и повторите разметку.')
        return row

    def create(self, job, parent, page, request):
        data = request.model_dump(mode='json')
        sha = hashlib.sha256(encoded(dict(parent=parent, page=page, request=data))).hexdigest()
        with self.queue.connect() as c:
            self._head(c, job, parent)
            old = c.execute('SELECT * FROM cad_tables_v1.proposals WHERE job_id=%s AND request_sha256=%s', (job, sha)).fetchone()
            if old and old['status'] not in ('cancelled', 'failed'): return old
            c.execute('SELECT pg_advisory_xact_lock(17441902)')
            if c.execute("SELECT count(*) AS n FROM cad_tables_v1.proposals WHERE status IN ('queued','running')").fetchone()['n'] >= 8:
                raise TableConflict('Очередь таблиц заполнена. Повторите позже.')
            if old:
                return c.execute("""UPDATE cad_tables_v1.proposals SET status='queued',attempts=0,error=NULL,
                    updated_at=clock_timestamp() WHERE id=%s RETURNING *""", (old['id'],)).fetchone()
            return c.execute("""INSERT INTO cad_tables_v1.proposals
                (id,job_id,parent_version,page_index,request,request_sha256,status)
                VALUES(%s,%s,%s,%s,%s,%s,'queued') RETURNING *""",
                (uuid4(), job, parent, page, self.queue.json(data), sha)).fetchone()

    def get(self, job, parent, page, identifier):
        with self.queue.connect() as c:
            row = c.execute('SELECT * FROM cad_tables_v1.proposals WHERE id=%s AND job_id=%s AND parent_version=%s AND page_index=%s',
                            (identifier, job, parent, page)).fetchone()
            if row is None: raise KeyError(identifier)
            return row

    def list(self, job, parent, page):
        with self.queue.connect() as c:
            return c.execute('SELECT * FROM cad_tables_v1.proposals WHERE job_id=%s AND parent_version=%s AND page_index=%s ORDER BY created_at DESC LIMIT 20',
                             (job, parent, page)).fetchall()

    def claim(self):
        with self.queue.connect() as c:
            c.execute("""UPDATE cad_tables_v1.proposals SET status='failed',owner=NULL,lease_until=NULL,error='attempts_exhausted'
                WHERE status='running' AND lease_until<=clock_timestamp() AND attempts>=3""")
            row = c.execute("""SELECT id FROM cad_tables_v1.proposals WHERE
                (status='queued' OR (status='running' AND lease_until<=clock_timestamp())) AND attempts<3
                ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1""").fetchone()
            if row is None: return None
            return c.execute("""UPDATE cad_tables_v1.proposals SET status='running',owner=%s,
                lease_until=clock_timestamp()+interval '90 seconds',attempts=attempts+1,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""", (uuid4(), row['id'])).fetchone()

    def heartbeat(self, row):
        with self.queue.connect() as c:
            return c.execute("""UPDATE cad_tables_v1.proposals SET lease_until=clock_timestamp()+interval '90 seconds'
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (row['id'], row['owner'])).rowcount == 1

    def finish(self, row, *, manifest=None, summary=None, error=None):
        status = 'failed' if error else ('prepared' if summary['has_changes'] else 'rejected')
        with self.queue.connect() as c:
            return c.execute("""UPDATE cad_tables_v1.proposals SET status=%s,owner=NULL,lease_until=NULL,
                preview_manifest=%s,summary=%s,error=%s,updated_at=clock_timestamp()
                WHERE id=%s AND owner=%s AND status='running' AND lease_until>clock_timestamp()""",
                (status, self.queue.json(manifest), self.queue.json(summary), error, row['id'], row['owner'])).rowcount == 1

    def cancel(self, row):
        with self.queue.connect() as c:
            return c.execute("""UPDATE cad_tables_v1.proposals SET status='cancelled',owner=NULL,lease_until=NULL,
                updated_at=clock_timestamp() WHERE id=%s AND status IN ('queued','running','prepared','rejected') RETURNING *""",
                (row['id'],)).fetchone()

    def publish(self, row, parent, approval, manifest, summary):
        with self.queue.connect() as c:
            c.execute('SELECT id FROM cad_service_v1.documents WHERE id=%s FOR UPDATE', (row['job_id'],))
            current = c.execute('SELECT * FROM cad_tables_v1.proposals WHERE id=%s FOR UPDATE', (row['id'],)).fetchone()
            if current['status'] == 'applied':
                if current['approval'] != approval: raise TableConflict('Результат уже принят с другой записью проверки.')
                return current['applied_version']
            doc = self._head(c, row['job_id'], row['parent_version'])
            if current['status'] != 'prepared' or current['preview_manifest'] != manifest:
                raise TableConflict('Нужен готовый неизменённый предпросмотр.')
            if approval['preview_sha256'] != manifest['sha256'] or not approval['reviewed']:
                raise TableConflict('Подтвердите точный предпросмотр.')
            live = c.execute('SELECT manifest FROM cad_service_v1.runs WHERE job_id=%s AND version=%s',
                             (row['job_id'], row['parent_version'])).fetchone()
            if live['manifest'] != parent['manifest']: raise TableConflict('Исходная версия изменилась.')
            version = doc['next_version']
            options = {**parent['options'], '_table_review': dict(id=str(row['id']), parent_version=row['parent_version'],
                page=row['page_index'], reviewer=row['request']['reviewer'], automatic_quality_credit=False)}
            c.execute('UPDATE cad_service_v1.documents SET next_version=next_version+1 WHERE id=%s', (row['job_id'],))
            c.execute("""INSERT INTO cad_service_v1.runs
                (id,job_id,version,options,status,manifest,summary,started_at,finished_at)
                VALUES(%s,%s,%s,%s,'completed',%s,%s,clock_timestamp(),clock_timestamp())""",
                (uuid4(), row['job_id'], version, self.queue.json(options), self.queue.json(manifest), self.queue.json(summary)))
            c.execute("""UPDATE cad_tables_v1.proposals SET status='applied',approval=%s,applied_version=%s,
                updated_at=clock_timestamp() WHERE id=%s""", (self.queue.json(approval), version, row['id']))
            return version
