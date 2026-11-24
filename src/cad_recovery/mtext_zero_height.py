"""Translate explicitly sized zero-base-height MTEXT without inventing a height.

This is a narrowly scoped, in-memory compatibility adapter for ezdxf 1.4.5.
The original MTEXT transform normalizes a zero-height vector even for a pure
translation. Here a proved translation changes only insertion and native XDATA.
Text, zero base height, width, directions and inline commands stay unchanged.
"""
from __future__ import annotations

import hashlib
import math

from ezdxf.entities import MText
from ezdxf.math import Matrix44
from ezdxf.tools.text import MTextContext, MTextParser, TokenType

METHOD = 'zero-base-mtext-explicit-height-translation-only-v1'
MAX_TEXT_CHARS = 65536
MAX_TOKENS = 10000
MAX_NATIVE_ENTITIES = 200000
MAX_TRANSLATION_RECORDS = 100000
_DOCUMENT_AUDIT = '_cad_recovery_zero_height_audit'


def _handle(entity):
    original = entity.origin_of_copy or entity
    return original.dxf.get('handle')


class UnsupportedZeroHeightMText(ValueError):
    """A zero-base-height entity is outside the explicitly supported scope."""

    def __init__(self, code, entity, message):
        self.code = code
        self.source_handle = _handle(entity)
        super().__init__(f'{code}: MTEXT {self.source_handle or "<virtual>"}: {message}')


def _reject(entity, code, message):
    raise UnsupportedZeroHeightMText(code, entity, message)


def _inert_columns(entity):
    if not entity.has_columns:
        return True
    columns = entity.columns
    return (int(columns.column_type) == 0 and columns.count == 1
            and columns.width == 0 and columns.defined_height == 0
            and columns.gutter_width == 0 and not columns.auto_height
            and not columns.reversed_column_flow and not columns.heights
            and not columns.linked_handles and not columns.linked_columns)


def validate_zero_height_mtext(entity) -> dict:
    """Prove rendered token sizes never depend on the invalid zero base height.

    A NaN initial context is a dependency marker. Relative heights cannot erase
    it; an explicit finite positive absolute height must precede every rendered
    token. The text parser and renderer share the installed ezdxf semantics.
    """
    if not isinstance(entity, MText) or entity.dxf.char_height != 0:
        _reject(entity, 'expected_zero_height_mtext', 'adapter requires native MTEXT char_height=0')
    text = entity.text
    if not isinstance(text, str) or not 0 < len(text) <= MAX_TEXT_CHARS:
        _reject(entity, 'unsupported_text_size', 'empty or oversized inline content')
    if entity.dxf.get('bg_fill', 0) or not _inert_columns(entity):
        _reject(entity, 'unsupported_zero_height_layout', 'background masks or active columns require separate proof')
    width = entity.dxf.get('width', 0)
    if not math.isfinite(width) or width < 0:
        _reject(entity, 'invalid_text_width', 'text width must be finite and nonnegative')
    direction = entity.get_text_direction()
    if (not all(math.isfinite(value) for value in (*direction, *entity.dxf.insert, *entity.dxf.extrusion))
            or direction.magnitude == 0 or abs(direction.z) > 1e-12
            or not entity.dxf.extrusion.isclose((0, 0, 1))):
        _reject(entity, 'unsupported_text_plane', 'only finite nondegenerate planar text is supported')
    ctx = MTextContext()
    ctx.cap_height = float('nan')
    sizes, command_count, token_count = [], 0, 0
    parser = MTextParser(text, ctx, yield_property_commands=True)
    for token in parser:
        token_count += 1
        if token_count > MAX_TOKENS:
            _reject(entity, 'text_token_budget', 'inline token count exceeds the adapter limit')
        if token.type == TokenType.PROPERTIES_CHANGED:
            if token.data.startswith('\\H'):
                if token.data[2:].startswith('-'):
                    _reject(entity, 'negative_inline_height', 'negative inline height is not normalized by this adapter')
                command_count += 1
            continue
        if token.type not in (TokenType.WORD, TokenType.SPACE, TokenType.NBSP, TokenType.STACK):
            _reject(entity, 'unsupported_layout_token', 'paragraphs, tabs, columns and wrapping controls require separate proof')
        paragraph = token.ctx.paragraph
        if paragraph.left or paragraph.right or paragraph.indent or paragraph.tab_stops:
            _reject(entity, 'base_dependent_paragraph_layout', 'paragraph offsets or tab stops depend on the base height')
        height = token.ctx.cap_height
        if not math.isfinite(height) or height <= 0:
            _reject(entity, 'default_dependent_token_height', 'each rendered token needs an explicit finite positive inline height')
        sizes.append(float(height))
    if not sizes or not command_count:
        _reject(entity, 'missing_explicit_text_size', 'no explicitly sized rendered content')
    return {'source_handle': _handle(entity), 'owner_handle': entity.dxf.get('owner'),
            'text_sha256': hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest(),
            'text_hash_encoding': 'UTF-8 surrogatepass; original code points are not corrected',
            'native_char_height': 0.0, 'native_width': float(width),
            'rendered_token_count': len(sizes), 'inline_height_command_count': command_count,
            'token_height_min': min(sizes), 'token_height_max': max(sizes),
            'token_heights_independent_of_default': True,
            'inert_column_metadata_present': bool(entity.has_columns),
            'renderer_width_estimation_unchanged': width < 1e-6}


class ZeroHeightMText(MText):
    """Keep the subclass on ezdxf virtual copies; never patch global classes."""

    def transform(self, matrix: Matrix44):
        if self.dxf.char_height != 0:
            return super().transform(matrix)
        proof = validate_zero_height_mtext(self)
        rows = [tuple(matrix.get_row(i)) for i in range(4)]
        if (not all(math.isfinite(value) for row in rows for value in row)
                or rows[:3] != [(1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.)]
                or rows[3][3] != 1.):
            _reject(self, 'nontranslation_zero_height_transform',
                    'rotation, scaling, reflection, shear, projection and nonfinite transforms are not supported')
        before = list(self.dxf.insert)
        after = matrix.transform(self.dxf.insert)
        if not all(math.isfinite(value) for value in after):
            _reject(self, 'nonfinite_translated_insert', 'translated insertion exceeds finite coordinates')
        audit = getattr(self.doc, _DOCUMENT_AUDIT, None) if self.doc is not None else None
        if audit is not None and len(audit['translations']) >= MAX_TRANSLATION_RECORDS:
            _reject(self, 'translation_record_budget', 'translation audit exceeds its record limit')
        # Do NOT assign dxf.char_height=0: its validating setter substitutes 2.5.
        self.dxf.insert = after
        self.post_transform(matrix)
        if audit is not None:
            audit['translations'].append({'source_handle': proof['source_handle'],
                'translation_xyz': list(rows[3][:3]), 'input_insert_xyz': before,
                'output_insert_xyz': list(after), 'char_height_preserved': 0.0,
                'inline_text_unchanged': True, 'width_unchanged': True})
            audit['translated_instances'] += 1
        return self


def install_zero_height_adapter(document) -> dict:
    """Validate native zero-height entities, then adapt only this loaded document.

    Validation completes before class substitution. Unused unsupported native
    entities are conservatively refused instead of silently skipped. The caller
    owns the temporary document and must not save it over the original DXF.
    """
    if len(document.entitydb) > MAX_NATIVE_ENTITIES:
        raise ValueError('Zero-height audit exceeds the native entity budget')
    if hasattr(document, _DOCUMENT_AUDIT):
        raise ValueError('Zero-height adapter is already installed in this document')
    prepared = []
    seen = set()
    for block in document.blocks:
        for entity in block:
            if entity.dxftype() == 'MTEXT' and entity.dxf.char_height == 0:
                identity = entity.dxf.handle
                if identity in seen:
                    raise ValueError('Repeated native MTEXT handle while installing adapter')
                seen.add(identity)
                proof = validate_zero_height_mtext(entity)
                proof['block_name'] = block.name
                prepared.append((entity, proof))
    report = {'schema_version': '1.0', 'method': METHOD,
        'native_zero_height_count': len(prepared), 'adapted_entity_count': len(prepared),
        'native_entities': [proof for _, proof in prepared], 'translated_instances': 0, 'translations': [],
        'base_height_changed': False, 'inline_text_changed': False, 'width_changed': False,
        'source_file_modified': False, 'temporary_document_classes_adapted': bool(prepared),
        'limitations': ['Only exact finite affine translations are supported for zero-base-height MTEXT.',
                       'Rotation and all scaling are refused; no inline heights are rewritten or rounded.',
                       'The renderer may estimate a width when native width is zero; existing behavior is retained, not verified against authoring CAD.',
                       'Native source handles and translation stages are recorded, not complete INSERT instance paths.',
                       'Text rendering compatibility is not semantic-label or physical-scale approval.']}
    setattr(document, _DOCUMENT_AUDIT, report)
    for entity, _ in prepared:
        entity.__class__ = ZeroHeightMText
    return report
