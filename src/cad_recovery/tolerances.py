"""Signed dimension deviations and the explicit legacy-v0.2 compatibility boundary.

The physical interval is [nominal + lower, nominal + upper]. DXF DIMTM stores
the negative of the lower deviation; it is not an absolute value.
"""
from dataclasses import dataclass
import math
from typing import Callable, Literal

ToleranceConvention = Literal['signed_deviations', 'legacy_magnitudes']


@dataclass(frozen=True)
class SignedDeviations:
    upper: float
    lower: float

    def dxf_style(self) -> dict:
        return {'dimtol': 1, 'dimtp': self.upper, 'dimtm': -self.lower if self.lower else 0., 'dimtdec': 6}

    def display_suffix(self, number: Callable[[float], str], unit: str = '') -> str:
        if self.upper == -self.lower:
            return '±' + number(self.upper) + unit

        def signed(value):
            # ezdxf retains a blank sign before a zero stacked deviation.
            sign = '+' if value > 0 else '-' if value < 0 else ' '
            return sign + number(abs(value)) + unit

        return signed(self.upper) + '^' + signed(self.lower)


def resolve_tolerances(nominal: float, upper: float | None, lower: float | None,
                       convention: ToleranceConvention = 'signed_deviations') -> SignedDeviations | None:
    if convention not in ('signed_deviations', 'legacy_magnitudes'):
        raise ValueError('Unknown tolerance convention')
    if upper is None and lower is None:
        return None
    upper = 0. if upper is None else float(upper)
    lower = 0. if lower is None else float(lower)
    if not all(math.isfinite(value) for value in (nominal, upper, lower)):
        raise ValueError('Tolerance interval must be finite')
    if convention == 'legacy_magnitudes':
        if upper < 0 or lower < 0:
            raise ValueError('Legacy tolerance magnitudes must be nonnegative')
        lower = -lower
    if upper < lower:
        raise ValueError('Tolerance upper deviation is below the lower deviation')
    if nominal <= 0 or nominal + lower <= 0 or not math.isfinite(nominal + upper):
        raise ValueError('Tolerance interval must remain positive and finite')
    return SignedDeviations(upper, lower)


def dimension_tolerances(dimension) -> SignedDeviations | None:
    """Resolve v0.1 signed fields or explicitly tagged v0.2 fields."""
    return resolve_tolerances(dimension.nominal, dimension.tolerance_plus, dimension.tolerance_minus,
                              getattr(dimension, 'tolerance_convention', 'signed_deviations'))
