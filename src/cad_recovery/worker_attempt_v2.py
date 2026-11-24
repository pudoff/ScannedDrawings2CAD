"""Supervised compute+seal and deadline-fenced SQL publication.

Only a small sealed receipt is read after the owned child stops. No output
hashing or artifact creation happens in the publishing parent.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from .runtime_worker_v1 import run_child, LeaseLost
from .runtime_io_v1 import atomic_json, read_json
from .worker_attempt_v1 import AttemptResult, record_failure
from .lease_keeper_v1 import LeaseKeeper


def execute_attempt(*, identity, request, folder, files, limits, script,
                    heartbeat, publish, memory_MiB=3072):
    folder = Path(folder); stage = 'prepare'
    keeper=LeaseKeeper(heartbeat,getattr(limits,'lease_seconds',90))
    def lease():
        try:
            return heartbeat()
        except Exception as error:
            # Ownership cannot be established during a DB outage. Fail closed;
            # do not attempt another SQL write over this uncertain lease.
            raise LeaseLost() from error
    try:
        atomic_json(folder/'attempt-context.json', dict(schema_version='worker-attempt/1.0',
            identity=identity.record(), state='computing'))
        keeper.start()
        stage = 'child'
        budget = run_child({**request, '_worker_identity': identity.record(),
                            '_artifact_storage': str(files.root)}, folder, files, limits, script,
                           keeper.alive, memory_MiB=memory_MiB)
        stage = 'seal'
        value = read_json(folder/'sealed-result.json', limit=16*1024*1024)
        if value['schema_version'] != 'sealed-child/1.0' or value['identity'] != identity.record():
            raise ValueError('Sealed child identity differs')
        manifest, summary = value['manifest'], value['summary']
        if (files.path(manifest['path']) != folder/'result-manifest.json'
                or not isinstance(manifest['sha256'], str) or len(manifest['sha256']) != 64):
            raise ValueError('Sealed manifest differs')
        stage = 'verify'
        budget.check()
        keeper.stop()
        # Renew immediately before the bounded publication transaction. The
        # child loop has kept the lease alive throughout compute AND seal.
        if not lease():
            raise LeaseLost()
        budget.check()
        atomic_json(folder/'attempt-context.json', dict(schema_version='worker-attempt/1.0',
            identity=identity.record(), state='ready_to_publish'))
        budget.finish('completed'); budget.check()
        remaining = limits.timeout_seconds-(time.monotonic()-budget.started)
        deadline = datetime.now(timezone.utc)+timedelta(seconds=remaining)
    except Exception as error:
        keeper.stop(strict=False)
        record_failure(files, identity, stage, error)
        raise
    # Once SQL may commit, do not modify this attempt or try to mark it failed.
    try:
        accepted = publish(manifest, summary, deadline)
    except Exception as error:
        record_failure(files, identity, 'publish', error, code='publication_uncertain')
        return AttemptResult('uncertain', manifest, summary)
    if not accepted:
        record_failure(files, identity, 'publish', LeaseLost())
        return AttemptResult('fenced', manifest, summary)
    return AttemptResult('published', manifest, summary)
