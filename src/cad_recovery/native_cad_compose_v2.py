"""Import DIMENSION blocks correctly and require native support at LINE seams.

Version1 remains frozen with its first integration protocol. This version
reuses it without duplicating the source-ownership/composition implementation.
"""
from collections import defaultdict
import json
from unittest.mock import patch

from ezdxf.addons import Importer

from . import native_cad_compose_v1 as previous
from .native_cad_stitch_v1 import stitch


class _CompoundObservations(dict):
    def __contains__(self, key):
        # Defer compound dedup until AFTER translation, including placement.
        return False if key[0] in ('DIMENSION', 'LEADER') else super().__contains__(key)


class Composer(previous.Composer):
    def __init__(self, size, source_sha):
        super().__init__(size, source_sha)
        self.compound = _CompoundObservations()

    def add(self, gray, tile, report):
        original = Importer.import_entity

        def bind(importer, entity, target_layout=None):
            layout = target_layout if target_layout is not None else importer.target.modelspace()
            before = len(layout.entity_space.entities)
            original(importer, entity, target_layout)
            if len(layout.entity_space.entities) == before+1:
                copied = layout.entity_space.entities[-1]
                if copied.dxftype() == 'DIMENSION':
                    # Importer copies graphical content virtually. Its normal
                    # entitydb insertion omits the Dimension binding hook.
                    # Bind the EXISTING content; do not recompute the dimension.
                    copied.post_bind_hook()
        with patch.object(Importer, 'import_entity', bind):
            return super().add(gray, tile, report)

    def finish(self, page, target):
        seen = {}
        for entity in list(self.model.query('DIMENSION LEADER')):
            attrs = {k: v for k, v in entity.dxf.all_existing_dxf_attribs().items()
                     if k not in ('handle', 'owner', 'geometry')}
            def normalize(value):
                if isinstance(value, float): return round(value, 9)
                if isinstance(value, (str, int, bool)) or value is None: return value
                return [normalize(v) for v in value]
            key = (entity.dxftype(), json.dumps({k: normalize(v) for k, v in attrs.items()}, sort_keys=True))
            if key in seen:
                old, new = entity.dxf.handle, seen[key]
                for uid, value in self.aliases.items():
                    if value == old: self.aliases[uid] = new
                self.model.delete_entity(entity)
            else: seen[key] = entity.dxf.handle

        def checked(rows, source):
            # Never merge CENTER observations with different dash scales.
            groups = defaultdict(list)
            for row in rows:
                entity = self.doc.entitydb[row['handle']]
                groups[(entity.dxf.linetype, entity.dxf.ltscale)].append(row)
            objects, aliases, comparisons, joined, passes, fragments = [], {}, 0, 0, 0, 0
            for (linetype, scale), values in groups.items():
                result, proof = stitch(values, source, CENTER_pattern=linetype.upper().startswith('CENTER'))
                objects.extend(result); aliases.update(proof['aliases'])
                comparisons += proof['comparisons']; joined += proof['joined_objects']
                passes = max(passes, proof['passes']); fragments += proof['unresolved_fragments']
            return objects, dict(aliases=aliases, comparisons=comparisons, joined_objects=joined,
                passes=passes, unresolved_fragments=fragments, whole_semantic_objects_accepted=False,
                continuous_lines_source_checked=True, distinct_CENTER_scales_kept=True)
        with patch.object(previous, 'stitch', checked):
            return super().finish(page, target)
