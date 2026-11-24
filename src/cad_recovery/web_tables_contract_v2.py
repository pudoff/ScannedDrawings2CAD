"""Separate binding for registered PDF pages and orthogonal table views."""
import hashlib
from .dataset_layout_v1 import encoded, digest, read
from .web_tables_contract_v1 import ROOT, Region, Request, Approval, parse_request

CODE = ('src/cad_recovery/table_registration_v1.py', 'src/cad_recovery/web_tables_context_v2.py',
    'src/cad_recovery/web_tables_build_v2.py', 'src/cad_recovery/web_tables_worker_v2.py',
    'src/cad_recovery/web_tables_api_v2.py', 'src/cad_recovery/web_tables_contract_v2.py',
    'scripts/build_web_tables_v2.py', 'scripts/run_profiles_v15.py')


def binding():
    import fitz
    from .web_tables_contract_v1 import binding as previous
    release = ROOT/'artifacts/web-tables-release-v1.json'
    if read(release)['decision'] != 'experimental_manual_aligned_raster_only': raise ValueError('Previous qualification changed')
    body = dict(schema_version='web-table-extension/2.0', mode='explicit_registered_page_review',
        previous=previous(), previous_qualification=dict(path=str(release.relative_to(ROOT)), sha256=digest(release)),
        code=[dict(path=f, sha256=digest(ROOT/f)) for f in CODE], pdf_renderer=fitz.VersionBind,
        automatic_routing=False, quality_accepted=False)
    return {**body, 'sha256': hashlib.sha256(encoded(body)).hexdigest()}
