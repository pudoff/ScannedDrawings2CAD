"""Frozen source producers and the complete inherited routes map; no target input."""
from pathlib import Path
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked
from cad_recovery.source_chain_profile_v2 import analyze as routes
from source_chain_inputs_v1 import prepare as previous_prepare


def prepare(root,view):
    gray,source,geometry,old,previous,local,heads,extra,calls=previous_prepare(root,view)
    if view.get('prior_routes'):
        inherited=np.array(Image.open(checked(Path(root),view['prior_routes']['roles'])))
        calls={**calls,'primary_routes_calls':0}
    else:
        inherited=routes(gray,previous,old,local['local']['text_mask'],heads,diagnostic_heads=extra)['pixel_roles']
        calls={**calls,'primary_routes_calls':1}
    return gray,source,geometry,inherited,previous,local,heads,extra,calls
