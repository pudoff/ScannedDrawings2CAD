"""Per-run monotone accounting shared by all supported chain stages."""
from dataclasses import dataclass, field


class ResourceExhausted(RuntimeError):
    """A bounded stage is unknown; it cannot publish a successful prefix."""


@dataclass(frozen=True)
class ResourceLimits:
    pixel_samples: int = 250_000
    search_checks: int = 2_000_000
    arrow_samples: int = 500_000
    detector_samples: int = 12_000_000

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise ValueError('Resource limits must be nonnegative integers')


@dataclass
class ResourceLedger:
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    used: dict = field(default_factory=dict, init=False)
    stages: dict = field(default_factory=dict, init=False)
    failures: list = field(default_factory=list, init=False)

    def charge(self, resource: str, amount: int, stage: str) -> None:
        if type(amount) is not int or amount < 0 or resource not in vars(self.limits):
            raise ValueError('Invalid resource charge')
        current = self.used.get(resource, 0)
        if current + amount > getattr(self.limits, resource):
            self.failures.append({'resource': resource, 'stage': stage, 'requested': amount,
                                  'used': current, 'limit': getattr(self.limits, resource)})
            raise ResourceExhausted(resource + ' exhausted in ' + stage)
        self.used[resource] = current + amount
        counts = self.stages.setdefault(stage, {})
        counts[resource] = counts.get(resource, 0) + amount

    def report(self) -> dict:
        from copy import deepcopy
        return {'version': 'chain-resources/1.0', 'limits': dict(vars(self.limits)),
                'used': dict(self.used), 'stages': deepcopy(self.stages),
                'complete': not self.failures, 'failures': deepcopy(self.failures),
                'scope': 'supported pixel session and reported/counted compatibility work; not all CPU or memory operations'}
