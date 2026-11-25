"""Exercise correction and table preview/publication workers on the smoke drawing."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main(base, timeout, job_id=None):
    def request(path, value=None):
        data = None if value is None else json.dumps(value).encode()
        headers = {'Content-Type': 'application/json', 'Origin': base}
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'{path}: {error.code} {error.read().decode()}') from error

    def wait(path, expected):
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() < deadline:
            value = request(path)
            if value['status'] != previous:
                previous = value['status']
                print(path.rsplit('/', 2)[0], previous, flush=True)
            if previous == expected:
                return value
            if previous in ('failed', 'stale', 'cancelled', 'rejected'):
                raise RuntimeError(json.dumps({k: value.get(k) for k in ('id', 'status', 'error')}))
            time.sleep(2)
        raise TimeoutError(path)

    if job_id is None:
        record = json.loads(Path('/app/artifacts/profile-service-v1/docker-smoke-result.json').read_text())
        job_path = record['status_url']
    else:
        from uuid import UUID
        job_path = '/v1/jobs/' + str(UUID(job_id))
    version = request(job_path)['latest_version']
    path = f'{job_path}/versions/{version}/pages/0/tables'
    value = request(path)
    assert value['permissions']['can_apply'], value['permissions']
    context = value['context']
    assert context['enabled'], context['reason']
    edit = {k: context[k] for k in ('context_sha256', 'source_manifest_sha256', 'source_sha256', 'binding_sha256')}
    # Explicitly mark the known 2x2 table of the generated synthetic page.
    edit.update(regions=[dict(id='smoke-table', kind='table', box_px=[350, 350, 610, 450])],
                page_kind='mixed', reviewer='Docker integration test')
    proposal = request(path + '/preview', edit)
    proposal_path = path + '/' + proposal['id']
    ready = wait(proposal_path, 'prepared')
    assert ready['preview_sha256'], ready
    applied = request(proposal_path + '/apply', dict(preview_sha256=ready['preview_sha256'], reviewed=True))
    assert request(job_path)['latest_version'] == applied['version']
    artifacts = request(f"{job_path}/versions/{applied['version']}/artifacts")['artifacts']
    assert any(a['filename'].endswith('.dxf') for a in artifacts)
    print('PASS: table preview and publication, version', applied['version'], flush=True)

    version = applied['version']
    page = f'{job_path}/versions/{version}/pages/0'
    path = page + '/corrections'
    context = request(path)
    assert context['permissions']['can_accept'], context['permissions']
    target = next(t for t in context['targets'] if t['kind'] == 'title_field' and t['id'] == 'name')
    proposal = request(path + '/preview', dict(source_manifest_sha256=context['source_manifest_sha256'],
        kind='title_field', target_id=target['id'], expected_old_value=target['value'],
        new_value='Docker smoke drawing ' + str(version), evidence_ref='Synthetic Docker integration test'))
    proposal_path = path + '/' + proposal['id']
    ready = wait(proposal_path, 'prepared')
    assert ready['preview_summary']['status'] == 'ready_for_review', ready
    request(proposal_path + '/accept', dict(proposal_sha256=ready['preview_summary']['proposal_sha256'],
        branch_reviewed=True, derived_dimensions_reviewed=True, review_ref='Synthetic Docker integration test'))
    applied = wait(proposal_path, 'applied')
    version = applied['applied_version']
    assert request(job_path)['latest_version'] == version
    print('PASS: correction preview and publication, version', version, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://web:8000')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--job-id', help='Existing smoke drawing; also allows running this test from the host')
    args = parser.parse_args()
    main(args.url.rstrip('/'), args.timeout, args.job_id)
