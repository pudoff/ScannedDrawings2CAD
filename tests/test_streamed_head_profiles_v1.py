import cv2
import numpy as np
import pytest

from cad_recovery.economical_head_profiles_v1 import Profiles as DenseProfiles
from cad_recovery.local_head_axis_profiles_v2 import finish as dense_finish
from cad_recovery.streamed_head_profiles_v1 import Profiles, finish
from cad_recovery import streamed_local_heads_v1 as detector


@pytest.mark.parametrize('angle,width,length',[(0,2,8),(17,7,18),(90,20,47),(147,60,100),
    (37,5.35,27.2),(113,31.17,72.8),(271,95,18.3),(180,3.5,44),(43,11,11),(92,50,80)])
def test_exact_width_and_centroid_against_frozen_dense_oracle(angle,width,length):
    rng=np.random.default_rng(3175);mask=np.zeros((256,256),np.uint8)
    for _ in range(32):
        a,b=rng.integers(0,256,(2,2));cv2.line(mask,tuple(a),tuple(b),255,int(rng.integers(1,12)))
    unit=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);args=np.array([127.31,125.87]),unit,length,width
    old,new=DenseProfiles(mask,eager=True),Profiles(mask)
    a,b=old.measure(*args,'body'),new.measure(*args,'body')
    reference=Profiles(mask,eager=True).measure(*args,'body')
    assert all(np.array_equal(x,y) for x,y in zip(reference,b))
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[3],b[3])
    assert np.allclose(a[1],b[1],atol=1e-4,rtol=0) and np.allclose(a[2],b[2],atol=1e-4,rtol=0)
    assert new.samples<=old.samples and not new.incomplete


def test_unknown_partial_span_requires_expansion_and_preserves_left_tie():
    across=np.arange(-8,8.25,.5);ink=np.zeros(len(across),bool)
    ink[(across>=-3.5)&(across<=-1)]=True;ink[(across>=2)&(across<=2.5)]=True
    lo=int(np.flatnonzero(across==-3)[0]);hi=int(np.flatnonzero(across==3)[0])+1
    status,_=Profiles.certificate(ink,across,lo,hi,3.)
    assert status=='expand'
    status,(a,b)=Profiles.certificate(ink,across,0,len(across),3.)
    assert status=='valid' and (across[a]+across[b-1])/2==-2.25


def test_far_unseen_spans_cannot_enter_the_allowed_center_band():
    mask=np.zeros((180,220),np.uint8);cv2.line(mask,(0,150),(219,150),255,3)
    args=np.array([60.,90.]),np.array([1.,0.]),30.,70.
    old,new=DenseProfiles(mask,eager=True),Profiles(mask)
    a,b=old.measure(*args,'body'),new.measure(*args,'body')
    assert np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2])
    assert new.samples<old.samples and new.statistics['rows_certified_zero']>0


def test_full_width_ink_has_no_false_claim_of_transverse_savings():
    mask=np.full((200,220),255,np.uint8);args=np.array([70.,90.]),np.array([1.,0.]),15.,8.
    old,new=DenseProfiles(mask,eager=True),Profiles(mask)
    a,b=old.measure(*args,'body'),new.measure(*args,'body')
    assert all(np.array_equal(x,y) for x,y in zip(a,b)) and new.samples==old.samples


def test_exact_row_reuse_does_not_recharge_later_stages():
    mask=np.zeros((200,220),np.uint8);cv2.line(mask,(10,90),(210,90),255,2)
    p=Profiles(mask);args=np.array([50.,90.]),np.array([1.,0.]),30.,12.
    p.measure(*args,'coarse');p.measure(*args,'tail');p.measure(*args,'body');spent=p.samples
    p.measure(*args,'coarse');p.measure(*args,'body')
    assert p.samples==spent and p.hits>0


def test_later_band_beyond_all_legal_starts_is_provably_irrelevant():
    mask=np.zeros((180,230),np.uint8);cv2.line(mask,(10,90),(50,90),255,2)
    cv2.fillConvexPoly(mask,np.array([[175,90],[120,80],[120,100]],np.int32),255)
    prepared=(np.array([50.,90.]),np.array([1.,0.]),60.,20.,(0,20,3.,10.))
    old,new=Profiles(mask,eager=True),Profiles(mask)
    assert finish(prepared,new,())==dense_finish(prepared,old,())
    assert new.samples<old.samples and new.statistics['body_positive_rows_unneeded']>0


def test_eligible_long_band_is_read_until_its_actual_end():
    mask=np.zeros((180,240),np.uint8);cv2.line(mask,(10,90),(40,90),255,2)
    cv2.fillConvexPoly(mask,np.array([[160,90],[40,80],[40,100]],np.int32),255)
    prepared=(np.array([40.,90.]),np.array([1.,0.]),60.,20.,(0,20,3.,10.))
    old,new=Profiles(mask,eager=True),Profiles(mask)
    assert finish(prepared,new,())==dense_finish(prepared,old,())
    entry=new.entry(*prepared[:4]);assert entry['body_certificate_index']>100


@pytest.mark.parametrize('width,angle',[(1,0),(1,37),(1,90),(2,0),(2,37),(2,90)])
def test_complete_heads_proposals_and_context_match_full_reference(width,angle):
    from test_sparse_local_heads_v1 import mask
    ink,_,_=mask(width,angle);a=detector.search(255-ink);b=detector.search(255-ink,eager_reference=True)
    assert not a['search_incomplete'] and not b['search_incomplete']
    for key in ('candidates','proposals','context_exclusions'):assert a[key]==b[key]


def test_budget_failure_does_not_mark_unknown_rows_observed():
    mask=np.full((100,100),255,np.uint8);p=Profiles(mask,maximum_samples=0)
    args=np.array([50.,50.]),np.array([1.,0.]),12.,8.
    assert p.measure(*args,'body') is None and p.incomplete and p.samples==0
    assert not p.entry(*args)['observed'].any()
