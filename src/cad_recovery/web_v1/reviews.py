"""Append-only human proposals. Automatic results and manifests are never edited."""
from uuid import uuid4
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from cad_recovery.service_v1.queue import Queue

class ReviewInput(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    page_index:int|None=Field(default=None,ge=0,le=19)
    field:Literal['comment','drawing_id','part_name','material','person','dimension']='comment'
    target:str=Field(default='',max_length=120)
    value:str=Field(default='',max_length=600)
    note:str=Field(default='',max_length=600)

    @model_validator(mode='after')
    def meaningful(self):
        self.target=self.target.strip();self.value=self.value.strip();self.note=self.note.strip()
        if any((ord(c)<32 and c not in '\t\r\n') or ord(c)==127 for text in (self.target,self.value,self.note) for c in text):
            raise ValueError('Unsupported control character')
        if self.field=='comment' and not self.note:raise ValueError('A comment must contain a note')
        if self.field!='comment' and not self.value:raise ValueError('A correction must contain a proposed value')
        return self

class Reviews:
    def __init__(self,settings):self.queue=Queue(settings)

    def initialize(self):
        with self.queue.connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(17441801)')
            c.execute("""CREATE SCHEMA IF NOT EXISTS cad_service_web_v1;
                CREATE TABLE IF NOT EXISTS cad_service_web_v1.reviews (
                id uuid PRIMARY KEY,job_id uuid NOT NULL,version integer NOT NULL,
                page_index integer CHECK(page_index BETWEEN 0 AND 19),field text NOT NULL,
                target text NOT NULL,value text NOT NULL,note text NOT NULL,
                source_manifest_sha256 text NOT NULL CHECK(length(source_manifest_sha256)=64),
                created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
                FOREIGN KEY(job_id,version) REFERENCES cad_service_v1.runs(job_id,version),
                CHECK(field IN ('comment','drawing_id','part_name','material','person','dimension')),
                CHECK(length(target)<=120 AND length(value)<=600 AND length(note)<=600));""")

    def documents(self,limit,offset,query):
        literal='%'+query.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
        with self.queue.connect() as c:
            total=c.execute('SELECT count(*) AS n FROM cad_service_v1.documents WHERE original_name ILIKE %s',(literal,)).fetchone()['n']
            rows=c.execute("""SELECT d.id,d.original_name,d.input_sha256,d.size_bytes,d.created_at,
                r.version AS latest_version,r.status,r.summary->>'status' AS quality
                FROM cad_service_v1.documents d JOIN LATERAL
                (SELECT version,status,summary FROM cad_service_v1.runs WHERE job_id=d.id ORDER BY version DESC LIMIT 1) r ON true
                WHERE d.original_name ILIKE %s ORDER BY d.created_at DESC,d.id DESC LIMIT %s OFFSET %s""",
                (literal,limit,offset)).fetchall()
        return {'documents':rows,'total':total,'next_offset':offset+len(rows) if offset+len(rows)<total else None}

    def list(self,job,version):
        self.queue.run(job,version)
        with self.queue.connect() as c:
            return c.execute('SELECT * FROM cad_service_web_v1.reviews WHERE job_id=%s AND version=%s ORDER BY created_at,id',(job,version)).fetchall()

    def add(self,job,version,review):
        run=self.queue.run(job,version)
        if run['status']!='completed' or not run['manifest']:raise LookupError('Review requires a completed result')
        if review.page_index is not None and review.page_index>=run['summary']['page_count']:
            raise ValueError('Review page is outside this document')
        with self.queue.connect() as c:
            return c.execute("""INSERT INTO cad_service_web_v1.reviews
                (id,job_id,version,page_index,field,target,value,note,source_manifest_sha256)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                (uuid4(),job,version,review.page_index,review.field,review.target,review.value,
                review.note,run['manifest']['sha256'])).fetchone()
