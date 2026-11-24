"""Read-only tracing of the sealed head detector; never proposes geometry.

The observer records gates actually executed, rather than reimplementing their
conditions. Labels belong to the subsequent evaluator, not this observer.
"""
import ast
from contextlib import contextmanager
import inspect
import sys

import numpy as np

from .economical_head_profiles_v1 import finish as fitted_finish
from .local_head_axis_profiles_v2 import prepare as root_prepare
from .local_head_axis_profiles_v1 import prepare as first_prepare
from .economical_head_profiles_v1 import prepare as initial_prepare
from .sparse_local_heads_v1 import runs


GATES = ('band_start_or_extent', 'falling_band_extent', 'fit_sample_count',
         'negative_taper', 'geometry_and_support', 'native_triangle_validation',
         'apex_source_ink')


def gate_locations():
    source, first = inspect.getsourcelines(fitted_finish)
    tree = ast.parse(''.join(source))
    nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and len(n.body) == 1 and isinstance(n.body[0], ast.Continue)),
                   key=lambda n: n.lineno)
    if len(nodes) != len(GATES):
        raise RuntimeError('The sealed fitter gate structure changed')
    return [{'name': name, 'start': first+n.lineno-1,
             'test_end': first+n.test.end_lineno-1,
             'continue': first+n.body[0].lineno-1,
             'end': first+n.end_lineno-1,
             'expression': ast.unparse(n.test)} for name, n in zip(GATES, nodes)]


def scalar_details(values, gate):
    result = {}
    fields = {
        'band_start_or_extent': ('local_width', 'extent'),
        'falling_band_extent': ('local_width', 'lo', 'hi'),
        'fit_sample_count': ('lo', 'hi'),
        'negative_taper': ('taper', 'intercept'),
        'geometry_and_support': ('local_width', 'error', 'root', 'base_width', 'head_length', 'support_ratio'),
        'native_triangle_validation': ('local_width', 'valid'),
        'apex_source_ink': ('local_width',),
    }
    for name in fields[gate]:
        value = values.get(name)
        if isinstance(value, (bool, np.bool_)):
            result[name] = bool(value)
        elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
            result[name] = float(value)
    if gate == 'band_start_or_extent':
        along, start, end = (values[k] for k in ('along', 'start', 'end'))
        result.update(band_start_px=float(along[start]), band_extent_px=float(along[end-1]-along[start]))
    if gate == 'fit_sample_count':
        result['fit_samples'] = int(values['chosen'].sum())
    if gate == 'native_triangle_validation' and isinstance(values.get('check'), dict):
        # This dictionary is already a JSON-safe native-validation report.
        from copy import deepcopy
        result['native_check'] = deepcopy(values['check'])
    return result


def shaft_details(values):
    """Describe already measured sections, including failed short source runs.

    These are explanatory values, never alternative acceptance conditions.
    No unobserved section is read or treated as measured source evidence.
    """
    if not all(k in values for k in ('along', 'widths', 'centers', 'width')):
        return None
    along, widths, centers = (values[k] for k in ('along', 'widths', 'centers'))
    options = []
    for a, b in runs((along < 0) & (widths > 0) & (widths <= values['width']*.8)):
        width = float(np.median(widths[a:b]))
        extent = float(along[b-1]-along[a]+.5)
        bridge = (along >= along[b-1]) & (along <= 0)
        centers_finite = np.isfinite(centers[a:b]).all()
        options.append({'extent_px': extent, 'required_extent_px': max(6., 2*width),
                        'width_px': width, 'last_station_px': float(along[b-1]),
                        'required_last_station_min_px': -max(3., 2*width),
                        'bridge_positive_fraction': float(np.mean(widths[bridge] > 0)),
                        'centers_finite': bool(centers_finite)})
    return {'runs': sorted(options, key=lambda x: -x['extent_px']),
            'section_queries_added': 0}


class GateObserver:
    def __init__(self, maximum_attempts=16384, maximum_gate_events=131072):
        if maximum_attempts < 1 or maximum_gate_events < 1:
            raise ValueError('Positive diagnostic bounds required')
        self.maximum_attempts = maximum_attempts
        self.maximum_gate_events = maximum_gate_events
        self.attempts = []
        self.gate_events = 0
        self.incomplete = False
        self.locations = gate_locations()
        self.active_prepare = None
        self.active_finish = None
        self.next_finish = None
        self.pending = None

    def gate(self, passed, values):
        if self.pending is None:
            return
        if self.gate_events >= self.maximum_gate_events:
            self.incomplete = True
        else:
            self.active_finish['body_gates'].append({
                'gate': self.pending['name'], 'passed': passed,
                **scalar_details(values, self.pending['name'])})
            self.gate_events += 1
        self.pending = None

    def trace(self, frame, event, arg):
        code, values = frame.f_code, frame.f_locals
        if event == 'call':
            if code is root_prepare.__code__:
                self.next_finish = None
                if len(self.attempts) >= self.maximum_attempts:
                    self.incomplete = True
                    self.active_prepare = None
                else:
                    record = {'seed': {k: np.asarray(values[k]).tolist()
                                       for k in ('base', 'unit', 'length', 'width')},
                              'prepare_chain': [], 'body_gates': []}
                    self.attempts.append(record)
                    self.active_prepare = record
                return self.trace
            if code in (initial_prepare.__code__, first_prepare.__code__):
                return self.trace if self.active_prepare is not None else None
            if code is fitted_finish.__code__:
                record = self.next_finish
                if record is None or record.get('prepare_reason') is not None:
                    return None
                self.active_finish = record
                return self.trace
            return None
        if code in (initial_prepare.__code__, first_prepare.__code__, root_prepare.__code__):
            if event == 'return' and self.active_prepare is not None and isinstance(arg, tuple):
                record = self.active_prepare
                item = {'module': frame.f_globals['__name__'], 'reason': arg[1]}
                if code is initial_prepare.__code__:
                    item['measured_shaft'] = shaft_details(values)
                record['prepare_chain'].append(item)
                if code is root_prepare.__code__:
                    prepared, reason = arg
                    record['prepare_reason'] = reason
                    self.next_finish = record if reason is None else None
                    if prepared is not None:
                        base, unit, length, width, shaft = prepared
                        record['prepared'] = {'base': base.tolist(), 'unit': unit.tolist(),
                            'length': float(length), 'width': float(width),
                            'shaft_width': float(shaft[2]), 'shaft_extent': float(shaft[3])}
                    self.active_prepare = None
            return self.trace
        if code is fitted_finish.__code__ and self.active_finish is not None:
            if event == 'line':
                line = frame.f_lineno
                if self.pending:
                    gate = self.pending
                    if line == gate['continue'] and line > gate['test_end']:
                        self.gate(False, values)
                    elif line < gate['start']:
                        self.gate(False, values)
                    elif line > gate['end']:
                        self.gate(True, values)
                if self.pending is None:
                    self.pending = next((g for g in self.locations
                                         if g['start'] <= line <= g['test_end']), None)
            elif event == 'return':
                if self.pending:
                    raise RuntimeError('Unresolved fitter branch')
                if isinstance(arg, tuple):
                    head, reason = arg
                    self.active_finish['finish_reason'] = reason
                    if head is not None:
                        from copy import deepcopy
                        self.active_finish['head'] = deepcopy(head)
                self.active_finish = None
            return self.trace
        return None

    @contextmanager
    def observing(self):
        previous = sys.gettrace()
        if previous is not None:
            raise RuntimeError('Run diagnostics outside another tracing session')
        sys.settrace(self.trace)
        try:
            yield self
        finally:
            sys.settrace(previous)

    def result(self):
        return {'attempts': self.attempts, 'gate_events': self.gate_events,
                'diagnostics_incomplete': self.incomplete,
                'gate_definitions': self.locations, 'predictor_parameters_changed': False,
                'labels_used_by_observer': False}
