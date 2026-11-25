"""Exercise real HTTP upload -> default CAD worker -> published, parseable DXF."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont
import ezdxf


def main(base, timeout):
    def request(path, data=None, content_type=None):
        headers = {'Content-Type': content_type} if content_type else {}
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=60) as response:
            return response.read()
    def get(path):
        return json.loads(request(path))

    assert get('/health')['status'] == 'ok'
    assert get('/v1/profiles')['default'] == 'native_cad_layers_draft'
    assert b'<html' in request('/').lower()
    image = Image.new('RGB', (640, 480), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 620, 460), outline='black', width=2)
    draw.rectangle((130, 120, 460, 310), outline='black', width=3)
    draw.ellipse((240, 170, 340, 270), outline='black', width=3)
    draw.line((130, 340, 460, 340), fill='black', width=2)
    draw.line((130, 315, 130, 355), fill='black', width=2)
    draw.line((460, 315, 460, 355), fill='black', width=2)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    draw.text((150, 55), 'ДЕТАЛЬ 120 ±2', fill='black', font=font)
    draw.text((270, 350), '120', fill='black', font=font)
    for x in (360, 480, 600):
        draw.line((x, 360, x, 440), fill='black', width=2)
    for y in (360, 400, 440):
        draw.line((360, y, 600, y), fill='black', width=2)
    for x, y, text in ((375, 363, 'A1'), (495, 363, 'B1'), (375, 403, 'A2'), (495, 403, 'B2')):
        draw.text((x, y), text, fill='black', font=font)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    job = json.loads(request('/v1/jobs?filename=docker-smoke.png', buffer.getvalue(), 'image/png'))
    job_path = job['status_url']
    print('Submitted:', job_path, flush=True)
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        status = get(job_path)
        if status['status'] != previous:
            previous = status['status']
            print('Status:', previous, flush=True)
        if previous == 'failed':
            raise RuntimeError(json.dumps(status['versions'][-1]['error']))
        if previous == 'completed':
            break
        time.sleep(3)
    else:
        raise TimeoutError('CAD conversion did not complete')
    version_path = job_path + '/versions/' + str(status['latest_version'])
    artifacts = get(version_path + '/artifacts')['artifacts']
    dxf = next(a for a in artifacts if a['filename'].endswith('.dxf'))
    data = request(dxf['download_url'])
    assert hashlib.sha256(data).hexdigest() == dxf['sha256']
    document = ezdxf.read(io.StringIO(data.decode('utf-8-sig')))
    assert len(document.modelspace()) > 0
    audit = document.audit()
    assert not audit.errors and not audit.fixes
    report_artifact = next(a for a in artifacts if a['filename'] == 'report.json')
    report = json.loads(request(report_artifact['download_url']))
    assert report['processing_profile']['name'] == 'native_cad_layers_draft'
    assert report['pages'][0]['demo_export'].get('font_file') == 'DejaVuSans.ttf', report['pages'][0]['demo_export']
    result = dict(job_id=job['job_id'], dxf_sha256=dxf['sha256'], entities=len(document.modelspace()),
                  status_url=job_path, download_url=dxf['download_url'])
    Path('/app/artifacts/profile-service-v1/docker-smoke-result.json').write_text(json.dumps(result, indent=2))
    print('PASS: default CAD profile, OCR, DXF publication/download and DXF audit.', flush=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://web:8000')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    main(args.url.rstrip('/'), args.timeout)
