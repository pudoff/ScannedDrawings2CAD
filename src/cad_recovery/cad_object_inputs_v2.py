"""Add precise graph subinterval ranges to the checked legacy observation codec."""
from copy import deepcopy
import json
from .cad_object_inputs_v1 import load_observations as previous
from .native_annotation_inputs_v1 import load_inputs as old_annotations
from .dataset_layout_v1 import checked


def enrich(root,packet,refs):
    graph=json.loads(checked(root,refs['graph.json']).read_text(encoding='utf8'));packet={**packet,'geometry':deepcopy(packet['geometry'])}
    for c in packet['geometry']:
        c['graph_interval_ranges']=[{'id':p['id'],'source_index_interval':p['source_index_interval']}
            for p in graph['proposals'] if p['parent_primitive_id']==c['id']]
    return packet


def load_observations(root,folder):
    packet,protection,refs,extra=previous(root,folder)
    return enrich(root,packet,refs),protection,refs,extra


def load_inputs(root,folder,ocr,**kwargs):
    packet,protected,annotations,refs,extra=old_annotations(root,folder,ocr,**kwargs)
    return enrich(root,packet,refs),protected,annotations,refs,extra
