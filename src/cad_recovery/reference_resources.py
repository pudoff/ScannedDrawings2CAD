"""Explicit resource completeness for reference rendering, without substitutions.

The native frontend otherwise draws filename/rectangle placeholders for missing
resources. Such output is useful diagnostically but cannot approve a pair.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path, PureWindowsPath

from PIL import Image
from ezdxf.addons.drawing.config import ImagePolicy

MAX_IMAGE_BYTES = 50 * 1024**2
MAX_IMAGE_PIXELS = 8_000_000
MAX_IMAGE_SIDE = 32768
MAX_DRAWN_RESOURCES = 4096


class IncompleteReferenceError(ValueError):
    """Required content is unavailable; ``issues`` explains the missing evidence."""

    def __init__(self, issues):
        self.issues = deepcopy(list(issues))
        self.reference_complete = False
        super().__init__('Reference is incomplete: '+', '.join(str(row['reason']) for row in self.issues))


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _network_path(value):
    return value.startswith(('\\\\', '//'))


class ResourceAwareFrontendMixin:
    """Guard only entities actually reached by the frontend's visibility walk.

    Call ``configure_resources`` after normal Frontend initialization. This
    mixin does not inspect unused blocks or make hidden entities visible.
    Diagnostic placeholders retain native placement but explicitly mark the
    reference incomplete. Native IMAGEDEF strings are never modified.
    """

    def configure_resources(self, source_path, allowed_root, allow_incomplete_reference=False):
        if not isinstance(allow_incomplete_reference, bool):
            raise ValueError('allow_incomplete_reference must be an explicit boolean')
        if _network_path(str(source_path)) or _network_path(str(allowed_root)):
            raise ValueError('Resource roots must be local filesystem paths')
        source, root = Path(source_path).resolve(), Path(allowed_root).resolve()
        if not source.is_relative_to(root) or not source.is_file() or not root.is_dir():
            raise ValueError('Source must be a file within the declared resource root')
        self._resource_source = source
        self._resource_root = root
        self._resource_source_sha = _sha(source)
        self._allow_incomplete_reference = allow_incomplete_reference
        self.resource_observations = []
        self.reference_complete = True

    def _resource_identity(self, entity):
        if not hasattr(self, '_resource_root'):
            raise ValueError('Configure reference resource handling before drawing')
        if len(self.resource_observations) >= MAX_DRAWN_RESOURCES:
            raise ValueError('Reference resource inventory exceeds its budget')
        native = entity.origin_of_copy or entity
        parents, visited = [], set()
        parent = entity.source_block_reference
        while parent is not None:
            if id(parent) in visited or len(parents) >= 32:
                raise ValueError('Reference resource INSERT provenance exceeds its budget')
            visited.add(id(parent))
            original = parent.origin_of_copy or parent
            parents.append(original.dxf.get('handle'))
            parent = parent.source_block_reference
        parents.reverse()
        if not parents:
            for parent in getattr(self, 'parent_stack', []):
                if parent.dxftype() == 'INSERT':
                    original = parent.origin_of_copy or parent
                    parents.append(original.dxf.get('handle'))
        return {'entity_type': entity.dxftype(), 'entity_handle': entity.dxf.get('handle'),
                'native_handle': native.dxf.get('handle'), 'native_owner_handle': native.dxf.get('owner'),
                'insert_path': parents, 'source_sha256': self._resource_source_sha,
                'loaded': False, 'resource_required': True, 'diagnostic_placeholder': False}

    def _incomplete(self, observation, reason, *, placeholder_supported=True):
        observation.update(reason=reason, status='incomplete_reference', reference_complete=False)
        self.reference_complete = False
        self.resource_observations.append(observation)
        if not self._allow_incomplete_reference or not placeholder_supported:
            raise IncompleteReferenceError([observation])
        observation['diagnostic_placeholder'] = True

    def _resolve_declared_image(self, filename):
        """Follow native local search order without opening an external path."""
        if not isinstance(filename, str) or not filename.strip():
            return None, 'empty_image_filename'
        if '\x00' in filename:
            return None, 'invalid_image_filename'
        if _network_path(filename):
            return None, 'external_network_image_path'
        windows = PureWindowsPath(filename)
        if windows.drive and not windows.is_absolute():
            return None, 'ambiguous_drive_relative_image_path'
        normalized = filename.replace('\\', os.sep)
        declared = Path(normalized)
        # A full external path is never rebound to a same-named local file.
        # Resolving an absolute local path does not read its file contents.
        if declared.is_absolute() and not declared.resolve().is_relative_to(self._resource_root):
            return None, 'image_path_outside_allowed_root'
        document_dir = getattr(self.ctx, 'document_dir', None)
        if document_dir is None or Path(document_dir).resolve() != self._resource_source.parent:
            return None, 'document_directory_differs_from_source'
        candidates = [declared, self._resource_source.parent/declared,
                      self._resource_source.parent/declared.name]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(self._resource_root):
                    # Native lookup considers cwd first for relative names.
                    # Detect a local shadowing file but never open/hash it.
                    if candidate.exists():
                        return None, 'image_path_outside_allowed_root'
                    continue
                if resolved.exists():
                    return (resolved, None) if resolved.is_file() else (None, 'image_path_is_not_a_file')
            except (OSError, ValueError):
                return None, 'image_path_unavailable'
        return None, 'image_file_missing'

    def _check_image(self, path):
        size = path.stat().st_size
        if not 0 < size <= MAX_IMAGE_BYTES:
            raise ValueError('image_file_size_exceeds_limit')
        before = _sha(path)
        with Image.open(path) as picture:
            width, height = picture.size
            if not (0 < width <= MAX_IMAGE_SIDE and 0 < height <= MAX_IMAGE_SIDE
                    and width*height <= MAX_IMAGE_PIXELS):
                raise ValueError('image_dimensions_exceed_limit')
            if getattr(picture, 'n_frames', 1) != 1:
                raise ValueError('animated_or_multipage_image_not_supported')
            metadata = {'size_bytes': size, 'size_px': [width, height], 'format': picture.format,
                        'sha256': before, 'path_relative_to_allowed_root': path.relative_to(self._resource_root).as_posix()}
            picture.verify()
        if path.stat().st_size != size or _sha(path) != before:
            raise ValueError('image_changed_during_preflight')
        return metadata

    def draw_image_entity(self, entity, properties):
        record = self._resource_identity(entity)
        definition = entity.image_def
        dimensions = [float(entity.dxf.image_size.x), float(entity.dxf.image_size.y)]
        record.update(image_def_handle=entity.dxf.get('image_def_handle'),
                      declared_filename=definition.dxf.get('filename', '') if definition is not None else None,
                      image_size_px=[value if math.isfinite(value) else None for value in dimensions],
                      show_image=bool(int(entity.dxf.flags) & 1))
        if definition is None:
            self._incomplete(record, 'missing_image_definition', placeholder_supported=False)
        if not all(math.isfinite(value) and value > 0 for value in dimensions):
            self._incomplete(record, 'invalid_image_dimensions', placeholder_supported=False)
        # No bitmap is requested when the native SHOW_IMAGE flag is off.
        # A valid IMAGEDEF is still required by the native boundary renderer.
        if not record['show_image']:
            record.update(status='bitmap_not_requested_by_entity_flags', reason='show_image_flag_disabled',
                          resource_required=False)
            self.resource_observations.append(record)
            return super().draw_image_entity(entity, properties)
        filename = record['declared_filename']
        path, reason = self._resolve_declared_image(filename)
        metadata = None
        if reason is None and self.config.image_policy != ImagePolicy.DISPLAY:
            reason = 'image_policy_does_not_display_native_content'
        if reason is None:
            try:
                metadata = self._check_image(path)
                if metadata['size_px'] != dimensions:
                    raise ValueError('image_dimensions_differ_from_declaration')
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                reason = 'image_validation_failed'
                record['validation_error'] = {'type': type(error).__name__, 'message': str(error)}
        if reason is not None:
            self._incomplete(record, reason)
            original_config = self.config
            self.config = replace(original_config, image_policy=ImagePolicy.MISSING)
            try:
                # This mode draws a labeled boundary; it never opens the image.
                return super().draw_image_entity(entity, properties)
            finally:
                self.config = original_config
        record['resource'] = metadata
        before_path, before_reason = self._resolve_declared_image(filename)
        if before_reason or before_path != path:
            self._incomplete(record, 'image_resolution_changed', placeholder_supported=False)
        super().draw_image_entity(entity, properties)
        after_path, after_reason = self._resolve_declared_image(filename)
        if (after_reason or after_path != path or path.stat().st_size != metadata['size_bytes']
                or _sha(path) != metadata['sha256']):
            self._incomplete(record, 'image_changed_during_render', placeholder_supported=False)
        record.update(status='loaded_pinned_resource', loaded=True, reason=None)
        self.resource_observations.append(record)

    def draw_ole2frame_entity(self, entity, properties):
        record = self._resource_identity(entity)
        self._incomplete(record, 'opaque_ole_content_not_rendered_by_native_frontend')
        # The original implementation draws only its geometric bounding box.
        # It remains available as a clearly incomplete diagnostic placeholder.
        return super().draw_ole2frame_entity(entity, properties)
