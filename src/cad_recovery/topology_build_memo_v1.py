"""Request-local memo of the complete deterministic topology build, not predicates."""
from collections import OrderedDict
from copy import deepcopy
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(',', ':')).encode('ascii')


class BuildMemo:
    """Reuse only exact inputs, preserving full results and logical QA counters.

    Each underlying build starts its own predicate budget. Cached results retain
    that complete build's counters/failure and never share an external budget.
    Instances belong to one isolated prediction and cannot be persisted or shared.
    """

    def __init__(self, build, *, maximum_entries=32, maximum_payload_bytes=8*1024**2, on_miss=None):
        self.build = build; self.values = OrderedDict(); self.on_miss = on_miss
        self.maximum_entries = maximum_entries; self.maximum_payload_bytes = maximum_payload_bytes
        self.payload_bytes = 0
        self.stats = dict(calls=0, hits=0, misses=0, bypassed=0, evicted=0,
            peak_entries=0, peak_serialized_payload_bytes=0)

    def __call__(self, candidates):
        self.stats['calls'] += 1
        try:
            key = canonical(candidates)
        except (TypeError, ValueError):
            self.stats['bypassed'] += 1
            return self.build(candidates)
        if key in self.values:
            self.stats['hits'] += 1
            graph, size = self.values.pop(key); self.values[key] = graph, size
            return deepcopy(graph)
        self.stats['misses'] += 1
        result = self.build(candidates)
        if canonical(candidates) != key:
            raise ValueError('Topology builder mutated its input; result cannot be memoized')
        if self.on_miss is not None:
            self.on_miss(candidates, result)
        size = len(key)+len(canonical(result))
        if size <= self.maximum_payload_bytes and self.maximum_entries > 0:
            self.values[key] = deepcopy(result), size; self.payload_bytes += size
            while len(self.values) > self.maximum_entries or self.payload_bytes > self.maximum_payload_bytes:
                _, (_, removed) = self.values.popitem(last=False)
                self.payload_bytes -= removed; self.stats['evicted'] += 1
            self.stats['peak_entries'] = max(self.stats['peak_entries'], len(self.values))
            self.stats['peak_serialized_payload_bytes'] = max(self.stats['peak_serialized_payload_bytes'], self.payload_bytes)
        return result
