"""Public worker correlation and failure contracts; no payload/path reflection."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID
import re

Stage = Literal['prepare', 'child', 'summary', 'verify', 'seal', 'publish']
STAGES = {'prepare', 'child', 'summary', 'verify', 'seal', 'publish'}


@dataclass(frozen=True)
class WorkerIdentity:
    request_id: str
    job_id: str
    run_id: str | None
    attempt_id: str
    version: int
    proposal_id: str | None = None
    page_index: int | None = None

    def __post_init__(self):
        for key in ('job_id', 'attempt_id'):
            if not _uuid(getattr(self, key)):
                raise ValueError('UUID identity required')
        if self.run_id is not None:
            if not _uuid(self.run_id):
                raise ValueError('UUID run identity required')
        elif self.proposal_id is None:
            raise ValueError('Run identity required for a profile attempt')
        if self.proposal_id is not None:
            if not _uuid(self.proposal_id):
                raise ValueError('UUID proposal identity required')
        if not isinstance(self.request_id, str) or not (re.fullmatch(r'[a-f0-9]{64}', self.request_id) or _uuid(self.request_id)):
            raise ValueError('UUID or SHA request identity required')
        if type(self.version) is not int or self.version < 1:
            raise ValueError('Positive version required')
        if self.page_index is not None and (type(self.page_index) is not int or not 0 <= self.page_index < 20):
            raise ValueError('Supported page identity required')

    def record(self):
        return asdict(self)


def _uuid(value):
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def identity_for_run(run):
    return WorkerIdentity(str(run['id']), str(run['job_id']), str(run['id']),
                          str(run['owner']), run['version'])


def identity_for_table(row, parent=None):
    return WorkerIdentity(row['request_sha256'], str(row['job_id']), str(parent['id']) if parent else None,
        str(row['owner']), row['parent_version'], str(row['id']), row['page_index'])


PUBLIC_CODES = {'processing_failed', 'table_processing_failed', 'processing_timeout', 'memory_limit',
                'output_limit', 'budget_monitor_failed', 'processing_incomplete', 'lease_lost_or_cancelled',
                'telemetry_persist_failed', 'input_integrity', 'profile_contract_changed', 'publication_uncertain'}


def public_failure_code(error, *, fallback='processing_failed'):
    code = getattr(error, 'code', None)
    return code if isinstance(code, str) and code in PUBLIC_CODES else fallback


def failure_record(identity: WorkerIdentity, stage: Stage, error: BaseException, *, code=None):
    if stage not in STAGES:
        raise ValueError('Known failure stage required')
    code = code or getattr(error, 'code', 'processing_failed')
    if not isinstance(code, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', code):
        code = 'processing_failed'
    return dict(schema_version='worker-failure/1.0', identity=identity.record(), stage=stage,
                code=code, error_type=type(error).__name__[:80])


class AttemptFiles(Protocol):
    root: Path

    def seal(self, folder: Path, summary: dict) -> dict: ...

    def output_size(self, folder: Path) -> tuple[int, int]: ...


class AttemptLimits(Protocol):
    timeout_seconds: float
    max_output_bytes: int
    max_output_files: int
