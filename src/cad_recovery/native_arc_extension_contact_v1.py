"""Conservative source-pixel footprints at an ARC/extension tangent contact.

These masks distinguish possible geometric footprints, not semantic owners.
In particular, extension_exclusive excludes this ARC only: all other source
owners, arrow heads, text, axes, and outline crossings remain caller-protected.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from .part_admission_v1 import object_sha


POLICY = {'version': 'native-arc-extension-contact/1.0',
    'maximum_roi_pixels': 4_000_000, 'maximum_samples': 500_000,
    'maximum_section_points': 256, 'minimum_sections_per_side': 8,
    'maximum_radial_profile_samples': 513, 'minimum_radial_side_fraction': .8,
    'minimum_source_fraction': .95, 'maximum_flank_angle_deg': 65.,
    'maximum_tangent_angle_deg': 8., 'pixel_radius_px': math.sqrt(.5),
    'semantic_roles_inferred': False, 'source_ownership_approved': False,
    'physical_geometry_approved': False}


def _point(value, label):
    point = np.asarray(value, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(label+' must be a finite source point')
    return point


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label+' must be finite and positive')
    return float(value)


def _sample(distance, points, arc_length, width):
    q = np.rint(points).astype(int)
    if ((q < 0).any() or (q[:,0] >= distance.shape[1]).any()
            or (q[:,1] >= distance.shape[0]).any()):
        return None, 'source_coverage_clipped_by_roi'
    seen = distance[q[:,1],q[:,0]] <= POLICY['pixel_radius_px']
    changes = np.diff(np.r_[False,~seen,False].astype(np.int8))
    gaps = np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
    gap = float(gaps.max(initial=0)*arc_length/max(1,len(points)-1))
    allowed = width+2*POLICY['pixel_radius_px']
    proof = {'samples':len(points), 'fraction':float(seen.mean()),
        'maximum_unexplained_gap_px':gap, 'allowed_unexplained_gap_px':allowed,
        'centerline_pixel_allowance_px':POLICY['pixel_radius_px']}
    return proof, None if proof['fraction'] >= POLICY['minimum_source_fraction'] and gap <= allowed else 'source_centerline_ink_missing'


def _radial_sections(ink, points, center, anchor, unit, width, offsets):
    counts=[0,0]; totals=[0,0]; widths=[]
    for point in points:
        side=int(float((point-anchor)@unit)>0); totals[side]+=1
        radial=point-center; norm=float(np.linalg.norm(radial))
        if norm<=0:
            return None,'invalid_arc_flank_point'
        radial/=norm
        q=np.rint(point+offsets[:,None]*radial).astype(int)
        if ((q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any()):
            return None,'radial_source_profile_clipped_by_roi'
        bits=ink[q[:,1],q[:,0]]
        change=np.diff(np.r_[False,bits,False].astype(np.int8))
        starts,ends=np.flatnonzero(change==1),np.flatnonzero(change==-1)
        if len(starts)!=1 or starts[0]==0 or ends[0]==len(offsets):
            continue
        observed_width=float(offsets[ends[0]-1]-offsets[starts[0]]+offsets[1]-offsets[0])
        if abs(observed_width-width)>max(2.,width*.25):
            continue
        counts[side]+=1; widths.append(observed_width)
    proof={'profiles_per_side':totals,'unambiguous_matching_width_profiles_per_side':counts,
        'measured_widths_px':widths,'minimum_side_fraction':POLICY['minimum_radial_side_fraction']}
    if any(count<POLICY['minimum_sections_per_side'] or count<POLICY['minimum_radial_side_fraction']*total
           for count,total in zip(counts,totals)):
        return proof,'arc_flank_profiles_not_confirmed'
    return proof,None


def observe(gray, extension, anchor, tip_px):
    """Return source-ink-only geometric masks and an auditable certificate.

    `complete` describes bounded computation, while report.accepted describes
    the source check. Every held observation returns empty masks. The original
    ARC localization uncertainty is preserved, including for contact pixels.
    """
    if not isinstance(gray,np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError('Nonempty uint8 source window required')
    # An over-budget input must not trigger three additional full allocations.
    empty = lambda: np.broadcast_to(np.array(False),gray.shape)
    report = {'schema_version':POLICY['version'], 'policy':deepcopy(POLICY),
        'status':'withheld', 'accepted':False, 'reasons':[], 'limits':[],
        'source_size_px':list(gray.shape[::-1]), 'source_pixels_sha256':None,
        'coordinate_space':'source_window_pixels', 'extension_sha256':object_sha(extension),
        'anchor_sha256':object_sha(anchor), 'tip_px':_point(tip_px,'tip').tolist(),
        'semantic_roles_inferred':False, 'source_ownership_approved':False,
        'physical_geometry_approved':False, 'source_pixels_changed':0,
        'source_interval_ownership_extended':False,
        'other_competing_owners_must_be_protected_by_caller':True,
        'curve_mask_scope':'full_circle_possible_footprint_conservative_beyond_observed_arc_domain',
        'extension_exclusive_scope':'source_ink_outside_this_arc_possible_band_only',
        'work':{'sample_count':0}, 'mask_counts':{}, 'mask_sha256':{}}

    def held(reason=None, limit=None):
        if reason: report['reasons'].append(reason)
        if limit: report['limits'].append(limit)
        return {'report':report,'shared':empty(),'extension_exclusive':empty(),
                'curve_possible':empty(),'complete':not report['limits']}

    if gray.size > POLICY['maximum_roi_pixels']:
        return held(limit='source_roi_pixel_budget')
    digest = hashlib.sha256(gray.tobytes()).hexdigest()
    report['source_pixels_sha256'] = digest
    reference = extension.get('source_profile',{}).get('input_pixels_sha256')
    if reference is not None and reference != digest:
        return held('extension_source_pixels_changed')
    evidence = anchor['evidence']
    if evidence.get('source_curve_geometry_confirmed') is not True or evidence.get('arc_domain_checked') is not True:
        return held('source_arc_observation_not_confirmed')
    center = _point(evidence['curve_center_px'],'curve center')
    radius = _positive(evidence['curve_radius_px'],'curve radius')
    arc_width = _positive(evidence['measured_curve_width_px'],'curve width')
    uncertainty = _positive(evidence['localization_uncertainty_px'],'curve localization uncertainty')
    input_uncertainty = _positive(evidence['input_uncertainty_px'],'input curve uncertainty')
    if uncertainty+1e-12 < input_uncertainty+POLICY['pixel_radius_px']:
        return held('curve_localization_bound_reduced_below_original_observation')
    extension_width = _positive(extension['width_px'],'extension width')
    extension_uncertainty = _positive(extension['uncertainty_px'],'extension uncertainty')
    half_angle = _positive(evidence['source_curve_coverage']['half_angle_deg'],'observed flank extent')
    if half_angle > POLICY['maximum_flank_angle_deg']:
        return held('unsupported_arc_observation_extent')
    point = _point(anchor['anchor_px'],'anchor')
    tip = _point(tip_px,'tip')
    ends = np.asarray(extension['points_px'],float)
    if ends.shape!=(2,2) or not np.isfinite(ends).all():
        raise ValueError('Finite extension endpoints required')
    length = float(np.linalg.norm(ends[1]-ends[0]))
    span = float(np.linalg.norm(tip-point))
    if length <= 0 or span <= 0:
        return held('degenerate_extension_or_contact_span')
    unit = (ends[1]-ends[0])/length
    radial = point-center
    radial_length = float(np.linalg.norm(radial))
    if radial_length <= 0 or abs(radial_length-radius) > POLICY['pixel_radius_px']:
        return held('anchor_not_on_observed_arc_extremum')
    if abs(float(unit@radial/radial_length)) > math.sin(math.radians(POLICY['maximum_tangent_angle_deg'])):
        return held('extension_not_tangent_to_observed_arc')
    normal = np.array([-unit[1],unit[0]])
    line_radius = extension_width/2+extension_uncertainty+POLICY['pixel_radius_px']
    for endpoint in (point,tip):
        along = float((endpoint-ends[0])@unit)
        if (along < -line_radius or along > length+line_radius
                or abs(float((endpoint-ends[0])@normal)) > line_radius):
            return held('contact_span_outside_observed_extension')
    # The mask corridor may not be implicitly trimmed by an ROI boundary.
    if any((p-line_radius<0).any() or (p+line_radius>=gray.shape[::-1]).any() for p in (point,tip)):
        return held('extension_footprint_clipped_by_roi')
    raw_sections=evidence.get('source_section_points_px',[])
    if len(raw_sections)>POLICY['maximum_section_points']:
        return held(limit='source_arc_section_budget')
    sections = np.asarray(raw_sections,float)
    if sections.ndim != 2 or sections.shape[1:]!=(2,) or not np.isfinite(sections).all():
        return held('invalid_source_arc_sections')
    offsets = (sections-point)@unit
    if min(np.count_nonzero(offsets<0),np.count_nonzero(offsets>0)) < POLICY['minimum_sections_per_side']:
        return held('two_sided_source_arc_sections_missing')
    theta = math.atan2(radial[1],radial[0])
    arc_length = 2*radius*math.radians(half_angle)
    arc_count = max(3,math.ceil(2*arc_length)+1)
    line_count = max(3,math.ceil(2*span)+1)
    # Rechecking a clean observed flank must not introduce a new, wider
    # radial search that captures a separate crossing outside its original
    # observation. This limits the verification window only; the full ARC
    # localization uncertainty remains in curve_possible below.
    original_radial_reach=math.ceil(input_uncertainty+3*extension_width+2)
    radial_reach=min(original_radial_reach,
        max(2*arc_width,arc_width/2+4*POLICY['pixel_radius_px']))
    report['radial_profile_reach_px']=radial_reach
    report['original_arc_observation_radial_reach_px']=original_radial_reach
    if radial_reach<=arc_width/2+POLICY['pixel_radius_px']:
        return held('observed_radial_profile_cannot_bracket_curve_width')
    radial_count=math.floor(4*radial_reach)+1
    if radial_count>POLICY['maximum_radial_profile_samples']:
        return held(limit='radial_source_profile_budget')
    total = arc_count+line_count+len(sections)*(radial_count+1)
    report['work']['sample_count']=total
    if total>POLICY['maximum_samples']:
        return held(limit='contact_source_sample_budget')
    ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
    distance = cv2.distanceTransform((~ink).astype(np.uint8)*255,cv2.DIST_L2,5)
    radial_check,reason=_radial_sections(ink,sections,center,point,unit,arc_width,
        np.linspace(-radial_reach,radial_reach,radial_count))
    report['radial_flank_recheck']=radial_check
    if reason:
        return held(reason)
    section_check, reason = _sample(distance,sections,0.,arc_width)
    report['source_section_recheck']=section_check
    if reason:
        return held('arc_sections_'+reason)
    angles = theta+np.linspace(-math.radians(half_angle),math.radians(half_angle),arc_count)
    curve = center+radius*np.column_stack((np.cos(angles),np.sin(angles)))
    curve_check, reason = _sample(distance,curve,arc_length,arc_width)
    report['arc_centerline_recheck']=curve_check
    if reason:
        return held('arc_'+reason)
    line = np.linspace(point,tip,line_count)
    line_check,reason = _sample(distance,line,span,extension_width)
    report['extension_centerline_recheck']=line_check
    if reason:
        return held('extension_'+reason)
    # Full-circle protection is intentionally broader than the observed arc.
    # It may withhold extra pixels but cannot grant them to the extension.
    curve_radius = arc_width/2+uncertainty+POLICY['pixel_radius_px']
    yy,xx = np.ogrid[:gray.shape[0],:gray.shape[1]]
    curve_possible = ink & (abs(np.hypot(xx-center[0],yy-center[1])-radius)<=curve_radius)
    along = (xx-point[0])*unit[0]+(yy-point[1])*unit[1]
    across = abs((xx-ends[0,0])*normal[0]+(yy-ends[0,1])*normal[1])
    bound = float((tip-point)@unit)
    extension_band = ink & (along>=min(0.,bound)) & (along<=max(0.,bound)) & (across<=line_radius)
    shared = extension_band & curve_possible
    exclusive = extension_band & ~curve_possible
    report.update(status='source_contact_observed',accepted=True,
        preserved_arc_localization_uncertainty_px=uncertainty,
        original_curve_uncertainty_px=input_uncertainty,
        curve_possible_band_radius_px=curve_radius,extension_band_radius_px=line_radius,
        extension_geometry_source_members=deepcopy(extension.get('source_members',[])),
        curve_geometry_source_members=deepcopy(anchor.get('source_members',[])),
        contact_span_px=[point.tolist(),tip.tolist()],
        localization_uncertainty_replaced_by_fit_residual=False)
    masks={'shared':shared,'extension_exclusive':exclusive,'curve_possible':curve_possible}
    report['mask_counts']={name:int(np.count_nonzero(mask)) for name,mask in masks.items()}
    report['mask_sha256']={name:hashlib.sha256(mask.tobytes()).hexdigest() for name,mask in masks.items()}
    return {'report':report,**masks,'complete':True}
