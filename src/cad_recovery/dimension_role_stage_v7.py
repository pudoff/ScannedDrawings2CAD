"""Persist source-bound body roles and an explicit optional automatic CV origin."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import dimension_role_stage_v6 as previous
from .dataset_layout_v1 import checked, pin
from .dimension_role_stage_v1 import write
from .grid_role_revision_v1 import cv_authority, sha
from .joint_dimension_roles_v6 import propose
from .source_arrow_observation_v1 import digest

ROOT = Path(__file__).resolve().parents[2]


def analyze(folder, packet, gray, roles, *, automatic_producer_code=None, **kwargs):
    folder = Path(folder).resolve()
    observed = {}
    protected = np.where(packet.get('protected', np.zeros_like(gray)) > 0, 255, 0).astype(np.uint8)

    def with_roles(image, **options):
        # The original stage creates its directory before invoking this callback.
        input_file = folder/'input-roles.png'
        Image.fromarray(roles).save(input_file)
        Image.fromarray(protected).save(folder/'input-review-protection.png')
        authority = None
        if automatic_producer_code is not None:
            code = [pin(ROOT, path) for path in automatic_producer_code]
            for record in code:
                checked(ROOT, record)
            producer = {'schema_version': 'automatic-dimension-role-input/1.0',
                'working_gray_sha256': sha(gray), 'roles_sha256': sha(roles),
                'original_source_sha256': digest(options['original_source']),
                'reviewed_input': False, 'code': code,
                'input_role_file': pin(ROOT, input_file.relative_to(ROOT).as_posix()),
                'scope': 'fresh_automatic_experimental_PNG_pipeline_with_explicit_review_protection'}
            write(folder/'automatic-role-origin.json', producer)
            authority = cv_authority(gray, roles,
                pin(ROOT, (folder/'automatic-role-origin.json').relative_to(ROOT).as_posix()), protected=protected)
        result = propose(image, **options, authority=authority, authority_root=ROOT, protected=protected)
        observed['native'] = result['native_roles']
        return result

    with patch.object(previous, 'propose', with_roles):
        result = previous.analyze(folder, packet, gray, roles, **kwargs)
    native = observed['native']
    write(folder/'native-dimension-roles.json', native['report'])
    Image.fromarray(native['proposed_roles']).save(folder/'proposed-native-roles.png')
    Image.fromarray(native['protection']).save(folder/'native-role-protection.png')
    return result
