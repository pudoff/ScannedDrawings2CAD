"""Public, opt-in c.13+c.14 pipeline. No service profile is modified."""
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol
from .chain_resources_v1 import ResourceLimits, ResourceLedger, ResourceExhausted
from .chain_legacy_bridge_v1 import GeometryBackend, demote
from .chain_verification_v1 import VerificationPolicy, VerificationSession
from .chain_ownership_v1 import OwnershipResolver
from .arrow_shaft_support_v1 import observe, association_view
from .scaled_arrow_detection_v2 import detect as scaled_detect
from .completeness_publication_v1 import gate


class ArrowDetector(Protocol):
    version: str

    def detect(self, source: Path, *, geometry: dict, ocr: dict, ledger: ResourceLedger) -> dict: ...


class ObservedArrowDetector:
    version = 'observed-arrow-adapter/1.0'

    def __init__(self, report):
        self.report = deepcopy(report)

    def detect(self, source, *, geometry, ocr, ledger):
        GeometryBackend(source, geometry, ocr, self.report)
        return deepcopy(self.report)


class ScaledArrowDetector:
    version = 'scaled-arrow-adapter/1.0'

    def detect(self, source, *, geometry, ocr, ledger):
        result = scaled_detect(source, geometry=geometry, text_polygons=[l['polygon_px'] for l in ocr['lines']])
        try:
            ledger.charge('detector_samples', result['analysis']['profile_samples'], 'arrow_detector')
        except ResourceExhausted:
            result['analysis']['search_limited'] = True
            result['issues'].append({'code':'arrow_search_limit'})
        return result


@dataclass(frozen=True)
class ChainPipelinePolicy:
    head_method: str = 'sampled'
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self):
        if self.head_method not in ('sampled', 'mask'):
            raise ValueError('Unknown head verification method')

    @property
    def preliminary(self):
        return VerificationPolicy('preliminary_verification', self.head_method, .65, require_bounded_face=False)

    @property
    def ownership(self):
        return VerificationPolicy('ownership_verification', self.head_method, .85)


def run(source_image, output, *, geometry, ocr, detector: ArrowDetector,
        policy=None, verifier_factory=VerificationSession):
    """Accept observations and explicit adapters, never expected labels/values.

    The caller owns the output directory. Capped results are retained as
    needs_review, and historical compatibility stages retain their own caps.
    """
    source = Path(source_image);output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy = policy or ChainPipelinePolicy()
    ledger = ResourceLedger(policy.limits)
    verifier = None;raw = None;support = None;heads = None
    preliminary_checks = []
    try:
        heads = detector.detect(source, geometry=geometry, ocr=ocr, ledger=ledger)
        GeometryBackend(source, geometry, ocr, heads)
        if ledger.failures: raise ResourceExhausted('detector budget incomplete')
        support = observe(source, geometry=geometry, ocr=ocr, arrows=heads)
        ledger.charge('arrow_samples', support['sample_count'], 'arrow_shaft_observation')
        view, view_heads = association_view(geometry, heads, support)
        backend = GeometryBackend(source, view, ocr, view_heads)
        baseline = backend.baseline(output/'geometry-baseline')
        ledger.charge('search_checks', baseline.get('search_checks', 0), 'compatibility_baseline')
        raw = backend.geometry_proposals(baseline)
        ledger.charge('search_checks', raw.get('search_checks', 0), 'compatibility_geometry')
        ledger.charge('pixel_samples', raw.get('pixel_samples', 0), 'compatibility_geometry')
        raw['contour_topology'] = raw.get('contour_topology') or baseline.get('contour_topology')
        verifier = verifier_factory(source, view, view_heads, ledger)
        verified = {s['arrow_id'] for s in support['spans']}
        for row in raw['associations']:
            if row['status'] != 'proposed': continue
            check = verifier.verify(row, verified, raw.get('contour_topology') or {'faces':[]}, policy.preliminary)
            preliminary_checks.append({'association_id':row['id'],'verification':check})
            row['evidence']['supported_pixel_verification'] = check
            if not check['valid']: demote(row, 'preliminary_pixel_support_unverified')
        raw['pixel_verification'] = {'complete':True,'events':preliminary_checks}
        resolver = OwnershipResolver(backend, verifier, ledger, policy.ownership)
        raw = resolver.resolve(raw, support)
        if not support['complete']: raw['complete'] = False
        spans = {s['id'] for s in support['spans']}
        for row in raw['associations']:
            for c in [row] + row.get('alternatives',[]):
                if spans.intersection(c['measured_candidate_ids']):
                    raise ValueError('Annotation support used as measured part geometry')
        for topology in (raw.get('contour_topology'), raw.get('text_chain_ownership',{}).get('contour_topology')):
            for face in (topology or {}).get('faces',[]):
                for edge in face['boundaries']:
                    if spans.intersection(edge['geometry_ids']):
                        raise ValueError('Annotation support used as part contour')
    except ResourceExhausted as exc:
        # Stages prior to association construction can also exhaust resources;
        # frame validation still runs on the ORIGINAL observation packet.
        if heads is None:
            common = {k:geometry[k] for k in ('source_sha256','processed_size_px')}
            heads = {**common,'candidates':[],'analysis':{'search_limited':True},'issues':[{'code':'arrow_search_limit'}]}
        frame = GeometryBackend(source, geometry, ocr, heads)
        if raw is None:
            raw = {'source_sha256':frame.digest,'processed_size_px':frame.size,
                   'associations':[], 'issues':[], 'history':[], 'status':'needs_review'}
        raw['complete'] = False
        raw.setdefault('issues',[]).append({'code':'joint_chain_resource_limit','message':str(exc)})
        for row in raw['associations']: demote(row,'joint_chain_resource_limit')
    raw['complete'] = bool(raw.get('complete')) and ledger.report()['complete']
    result = gate(source, geometry=geometry, ocr=ocr, arrows=heads, proposals=raw)
    return {'version':'supported-chain-pipeline/1.0','status':'needs_review',
            'detector_version':detector.version,'arrows':heads,'support':support,
            'raw':raw,'gated':result,'resources':ledger.report(),
            'verification_session':verifier.report() if verifier else None,
            'verification_policies':{'preliminary':asdict(policy.preliminary),'ownership':asdict(policy.ownership)},
            'original_geometry_preserved':deepcopy(geometry),'original_ocr_preserved':deepcopy(ocr),
            'production_admitted':False,'physical_geometry_approved':False,'service_changed':False}
