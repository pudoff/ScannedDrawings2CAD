"""Quarantine an ambiguous train foot target without giving a model success credit."""
from pathlib import Path
from cad_recovery.dataset_layout_v1 import checked, pin
from capture_reference_context_inputs_v1 import CONTROLS
from run_topology_robust_v1 import read, write

ROOT=Path(__file__).resolve().parents[1]


def build():
    contract=read(checked(ROOT,CONTROLS))
    cases=[j['case']['id'] for j in contract['jobs'] if j['case']['labels']['family']=='double_level_part']
    assert len(cases)==4 and all(j['case']['split']=='train' for j in contract['jobs'] if j['case']['id'] in cases)
    write(ROOT/'artifacts/reference-context-qualification-v1/train-label-qa.json',{
       'controls':CONTROLS,'reviewer':'assistant_source_recipe_review','human_verified':False,
       'source_geometry_reason':'The added vertical part wall from [190,285] to [190,345] overlaps the extension. Both endpoints lie on the same datum plane; the rendered image does not identify [190,345] as a unique contact foot.',
       'cases_with_presence_but_no_unique_foot':cases,'chain_presence_known':True,
       'original_heads_roles_feet_and_predictions_changed':False,
       'strict_original_point_scores_retained':True,'ambiguous_proposals_get_TP_credit':False,
       'fresh_validation_labels_changed':False,'inference_receives_this_review':False,
       'policy':'Keep strict fixed-point scores as diagnostics. Exclude this entire train family from usable foot-regression scores; retain presence/heads/pixels. Fresh validation and all explicit negatives retain their original targets.',
       'generator':pin(ROOT,'scripts/review_reference_context_labels_v1.py')})
    print(pin(ROOT,'artifacts/reference-context-qualification-v1/train-label-qa.json'))


if __name__=='__main__':build()
