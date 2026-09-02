"""Final-answer extraction and exact numeric normalization."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER = r"[-+]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\s*\d+)?"
_MARKED_PATTERNS = (
    re.compile(rf"####\s*({_NUMBER})", re.IGNORECASE),
    re.compile(rf"\\boxed\s*\{{\s*({_NUMBER})\s*\}}", re.IGNORECASE),
    re.compile(
        rf"(?:final\s+answer|the\s+answer|answer)\s*(?:is|=|:)?\s*\$?\s*({_NUMBER})",
        re.IGNORECASE,
    ),
)
_ANY_NUMBER = re.compile(_NUMBER)
_LATEX_FRACTION = re.compile(
    r"\\frac\s*\{\s*([-+]?\s*\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\s*\d+(?:\.\d+)?)\s*\}"
)


def _strip_numeric_text(value: str) -> str:
    value = value.strip().replace("−", "-").replace("–", "-")
    value = value.replace(",", "").replace("$", "")
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".。!?,;:")


def normalize_answer(value: object) -> str | None:
    """Return a canonical exact rational representation, or ``None``.

    Decimal strings are converted through :class:`Decimal`, so values such as
    ``5``, ``5.0``, and ``10/2`` compare equal without floating-point error.
    """

    if value is None:
        return None
    text = _strip_numeric_text(str(value))
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", maxsplit=1)
            fraction = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            fraction = Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)


def extract_final_answer(text: object) -> str | None:
    """Extract and normalize the most likely final numeric answer.

    Explicit markers win. If no marker is present, the last numeric expression
    is used, matching the common convention for free-form math generations.
    """

    if text is None:
        return None
    rendered = _LATEX_FRACTION.sub(lambda match: f"{match.group(1)}/{match.group(2)}", str(text))
    for pattern in _MARKED_PATTERNS:
        matches = pattern.findall(rendered)
        if matches:
            return normalize_answer(matches[-1])
    matches = _ANY_NUMBER.findall(rendered)
    return normalize_answer(matches[-1]) if matches else None
