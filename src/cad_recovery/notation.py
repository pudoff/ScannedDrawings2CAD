"""Conservative syntax proposals for engineering OCR; never accepted dimensions."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
import re


MAX_INPUT_LENGTH = 512
MAX_VALUE = Decimal("1000000000")
MIN_NONZERO = Decimal("0.000000001")
_NUMBER = r"(?:[0-9]{1,12}(?:[.,][0-9]{1,9})?|[.,][0-9]{1,9})"
_SIGNED = rf"[+-]\s*{_NUMBER}"
_FIT = r"(?:JS|js|[A-HJKMNPRSTUVXYZ]|[a-hjkmnprstuvxyz])(?:01|0|1[0-8]|[1-9])"
_UNITS = r"(?:\s*(?:mm|мм))?"
_TRANSLATION = str.maketrans({"−": "-", "∅": "Ø", "⌀": "Ø", "﹢": "+", "﹣": "-"})


def _number(value: str, *, positive: bool = False) -> Decimal:
    try:
        result = Decimal(value.replace(" ", "").replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError("Invalid number") from exc
    if not result.is_finite() or abs(result) > MAX_VALUE or (result and abs(result) < MIN_NONZERO):
        raise ValueError("Number outside supported bounds")
    if positive and result <= 0:
        raise ValueError("Expected a positive number")
    return result


def parse_notation(raw: str) -> dict:
    """Parse a whole OCR string, retaining spelling and unsupported forms.

    Returned ``values`` are JSON numbers, computed with Decimal. ``parsed`` means
    syntax recognized only: position, units, OCR accuracy and geometry are not
    established here. Tolerance values are signed upper and lower deviations.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    result = {"raw": raw, "normalized": raw, "status": "unparsed", "kind": None, "values": {}, "issues": []}

    def fail(code: str, message: str, *, kind=None) -> dict:
        result.update(status="ambiguous", kind=kind, values={})
        result["issues"].append({"code": code, "message": message})
        return result

    def success(kind: str, **values) -> dict:
        result.update(status="parsed", kind=kind, values={
            key: float(value) if isinstance(value, Decimal) else value for key, value in values.items()
        })
        return result

    if len(raw) > MAX_INPUT_LENGTH:
        return fail("notation_too_long", "Input exceeds 512 characters.")
    normalized = " ".join(raw.translate(_TRANSLATION).split())
    result["normalized"] = normalized
    if not normalized:
        return result
    if any(char in raw for char in "\n\r\v\f"):
        return fail("multiline_notation", "Line layout is required to interpret stacked text.")
    if any(ord(char) < 32 and char != "\t" for char in raw):
        return fail("control_character", "Control characters are not valid notation.")

    try:
        # Explicit context such as material names, GOST and document identifiers
        # cannot match these anchored grammars, even when they contain numbers.
        match = re.fullmatch(rf"(?:(?:масштаб\s*:?|[MМ])\s*)?({_NUMBER})\s*:\s*({_NUMBER})", normalized, re.I)
        if match:
            numerator, denominator = (_number(value, positive=True) for value in match.groups())
            with localcontext() as context:
                context.prec = 32
                ratio = denominator / numerator
            if not MIN_NONZERO <= ratio <= MAX_VALUE:
                raise ValueError("Scale ratio outside supported bounds")
            return success("scale", scale_numerator=numerator, scale_denominator=denominator, model_per_paper=ratio)

        match = re.fullmatch(rf"({_NUMBER})\s*°", normalized)
        if match:
            angle = _number(match[1], positive=True)
            if angle > 360:
                raise ValueError("Angle exceeds 360 degrees")
            return success("angle", angle_deg=angle)

        match = re.fullmatch(rf"({_NUMBER})\s*[xх×]\s*({_NUMBER})\s*°", normalized)
        if match:
            distance, angle = (_number(value, positive=True) for value in match.groups())
            if angle >= 180:
                raise ValueError("Chamfer angle must be less than 180 degrees")
            return success("chamfer", length_mm=distance, angle_deg=angle)

        match = re.fullmatch(rf"[MМ]\s*({_NUMBER})(?:\s*[xх×]\s*({_NUMBER}))?(?:\s*-\s*([4-8][gGhH]))?", normalized)
        if match:
            diameter = _number(match[1], positive=True)
            values = {"diameter_mm": diameter, "thread_system": "metric", "pitch_mm": None, "fit": match[3]}
            if match[2] is not None:
                pitch = _number(match[2], positive=True)
                if pitch >= diameter:
                    raise ValueError("Thread pitch must be smaller than diameter")
                values["pitch_mm"] = pitch
            return success("thread", **values)

        count = None
        body = normalized
        # A quantity is recognized only together with an explicit diameter sign.
        match = re.fullmatch(r"([0-9]{1,5})\s*(?:[xх×]|отв\.?)\s*(Ø.+)", body)
        if match:
            count = int(match[1])
            if not 1 <= count <= 10000:
                raise ValueError("Hole count outside supported bounds")
            body = match[2]

        match = re.fullmatch(
            rf"(?P<prefix>Ø|R)?\s*(?P<nominal>{_NUMBER})"
            rf"(?:\s*(?P<fit>{_FIT})|\s*±\s*(?P<sym>{_NUMBER})|"
            rf"\s*(?P<upper>{_SIGNED})(?:\s*/\s*|\s+)(?P<lower>{_SIGNED}))?{_UNITS}", body)
        if match:
            if match["prefix"] is None and re.match(r"0[0-9]", match["nominal"]):
                return fail("leading_zero_ambiguous", "Leading zero may be a misread diameter sign or identifier; retained without guessing.")
            nominal = _number(match["nominal"], positive=True)
            kind = {"Ø": "diameter", "R": "radius", None: "nominal"}[match["prefix"]]
            values = {"nominal_mm": nominal}
            if kind != "nominal":
                values[f"{kind}_mm"] = nominal
            if count is not None:
                values["count"] = count
            if match["fit"]:
                values["fit"] = match["fit"]
            if match["sym"] is not None:
                deviation = _number(match["sym"])
                if nominal - deviation <= 0:
                    raise ValueError("Tolerance interval must remain positive")
                values.update(tolerance_plus_mm=deviation, tolerance_minus_mm=-deviation)
            elif match["upper"] is not None:
                upper, lower = _number(match["upper"]), _number(match["lower"])
                if upper < lower or nominal + lower <= 0:
                    raise ValueError("Tolerance bounds are reversed or nonpositive")
                values.update(tolerance_plus_mm=upper, tolerance_minus_mm=lower)
            return success(kind, **values)

        if re.fullmatch(rf"{_NUMBER}\s*[xх×]\s*{_NUMBER}", normalized):
            return fail("multiplication_without_context", "Could denote a size, quantity or chamfer; context is missing.")
        if re.fullmatch(r"(?:Ø|R|[MМ])?[0-9OoОо., +−\-/±:xх×°]+", normalized) and any(char in normalized for char in "OoОо"):
            return fail("ocr_letter_digit_confusion", "Letter O is retained; replacing it with zero requires review.")
        return result
    except ValueError as exc:
        return fail("notation_value_out_of_range", str(exc))
