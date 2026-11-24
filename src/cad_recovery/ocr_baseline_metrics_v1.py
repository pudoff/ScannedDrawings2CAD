"""Fixed, code-point OCR diagnostics for development line crops.

Raw predictions are never repaired. The separate normalized comparison uses a
small, declared symmetric mapping. Levenshtein ties are resolved while tracing
back from the end: match/substitution, then deletion, then insertion. Engineering
counts describe this one deterministic optimal alignment, not causal attribution.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
import unicodedata


ENGINEERING_CHARACTERS_V1 = ("⌀", "×", "°", "–", ",", "*", ".", "R", "M", "H", "h")
COMPARISON_POLICY_V1 = {
    "id": "ocr-development-comparison/1.0",
    "unit": "Unicode code point",
    "raw": "unchanged reference and assembled prediction",
    "normalization": {
        "unicode": "NFC",
        "whitespace": "collapse Unicode whitespace runs to ASCII space and trim",
        "character_map": {"Ø": "⌀", "∅": "⌀", "-": "–", "−": "–"},
        "application": "symmetrically to reference and prediction",
        "case_folding": False,
        "decimal_separator_conversion": False,
        "latin_cyrillic_conversion": False,
        "x_to_multiplication_conversion": False,
        "punctuation_removal": False,
    },
    "alignment": "unit-cost Levenshtein; traceback from end; diagonal, deletion, insertion tie order",
    "cer": "sum of substitutions, deletions and insertions / sum of reference code points",
    "exact_match": "identical complete strings / all input lines, including empty/error predictions",
    "empty_and_error_prediction": "empty string; all reference code points count as deletions",
    "engineering_characters": list(ENGINEERING_CHARACTERS_V1),
    "engineering_errors": "aligned non-match involving an engineering reference or prediction character",
    "engineering_limitation": "alignment-dependent diagnostics, not independent symbol detection accuracy",
}


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(name + " must be a string")
    return value


def normalize_text_v1(text: str) -> str:
    """Apply only the declared comparison policy, without using a target label."""
    value = unicodedata.normalize("NFC", _text(text, "text"))
    value = value.translate(str.maketrans(COMPARISON_POLICY_V1["normalization"]["character_map"]))
    return re.sub(r"\s+", " ", value).strip()


def _engineering(alignment):
    symbols = set(ENGINEERING_CHARACTERS_V1)
    references = {c: {"occurrences": 0, "matches": 0, "substitutions": 0, "deletions": 0}
                  for c in ENGINEERING_CHARACTERS_V1}
    predictions = {c: {"occurrences": 0, "matches": 0, "substitutions": 0, "insertions": 0}
                   for c in ENGINEERING_CHARACTERS_V1}
    errors = Counter()
    confusions = Counter()
    for item in alignment:
        ref, hyp, operation = item["reference"], item["prediction"], item["operation"]
        if ref in symbols:
            references[ref]["occurrences"] += 1
            references[ref][{"match": "matches", "substitution": "substitutions", "deletion": "deletions"}[operation]] += 1
        if hyp in symbols:
            predictions[hyp]["occurrences"] += 1
            predictions[hyp][{"match": "matches", "substitution": "substitutions", "insertion": "insertions"}[operation]] += 1
        if operation != "match" and (ref in symbols or hyp in symbols):
            errors[operation] += 1
            confusions[operation, ref, hyp] += 1
    return {
        "reference_occurrences": sum(v["occurrences"] for v in references.values()),
        "prediction_occurrences": sum(v["occurrences"] for v in predictions.values()),
        "substitutions": errors["substitution"],
        "deletions": errors["deletion"],
        "insertions": errors["insertion"],
        "error_events": sum(errors.values()),
        "by_reference_character": references,
        "by_prediction_character": predictions,
        "confusions": [
            {"operation": op, "reference": ref, "prediction": hyp, "count": count}
            for (op, ref, hyp), count in sorted(confusions.items(), key=lambda item: tuple(v or "" for v in item[0]))
        ],
    }


def score_text_v1(reference: str, prediction: str) -> dict:
    """Score raw code points and retain one complete deterministic alignment."""
    reference = _text(reference, "reference")
    prediction = _text(prediction, "prediction")
    n, m = len(reference), len(prediction)
    distances = [list(range(m + 1))]
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(prediction, 1):
            row.append(min(distances[i - 1][j - 1] + (ref != hyp),
                           distances[i - 1][j] + 1, row[j - 1] + 1))
        distances.append(row)
    alignment = []
    i, j = n, m
    while i or j:
        if i and j and distances[i][j] == distances[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            operation = "match" if reference[i - 1] == prediction[j - 1] else "substitution"
            alignment.append({"operation": operation, "reference": reference[i - 1], "prediction": prediction[j - 1],
                              "reference_index": i - 1, "prediction_index": j - 1})
            i, j = i - 1, j - 1
        elif i and distances[i][j] == distances[i - 1][j] + 1:
            alignment.append({"operation": "deletion", "reference": reference[i - 1], "prediction": None,
                              "reference_index": i - 1, "prediction_index": None})
            i -= 1
        else:
            alignment.append({"operation": "insertion", "reference": None, "prediction": prediction[j - 1],
                              "reference_index": None, "prediction_index": j - 1})
            j -= 1
    alignment.reverse()
    counts = Counter(item["operation"] for item in alignment)
    return {
        "reference": reference, "prediction": prediction,
        "reference_characters": n, "prediction_characters": m,
        "substitutions": counts["substitution"], "deletions": counts["deletion"], "insertions": counts["insertion"],
        "distance": distances[n][m], "cer": distances[n][m] / n if n else None,
        "exact_match": reference == prediction, "alignment": alignment,
        "engineering": _engineering(alignment),
    }


def _summary(scores):
    keys = ("reference_characters", "prediction_characters", "substitutions", "deletions", "insertions", "distance")
    totals = {key: sum(score[key] for score in scores) for key in keys}
    exact = sum(score["exact_match"] for score in scores)
    return {
        "sample_count": len(scores), **totals,
        "cer": totals["distance"] / totals["reference_characters"] if totals["reference_characters"] else None,
        "exact_matches": exact, "exact_match_rate": exact / len(scores) if scores else None,
        "engineering": _engineering(item for score in scores for item in score["alignment"]),
    }


def _status_counts(rows):
    return {status: sum(row["status"] == status for row in rows) for status in ("ok", "empty", "error")}


def aggregate_scores_v1(rows: list[dict]) -> dict:
    """Score a complete line registry; no failed or empty prediction is dropped.

    Corpus references must contain a non-whitespace target. Individual empty
    references remain supported by score_text_v1 for analytical controls.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    seen, samples = set(), []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "source_id", "reference", "prediction", "status"}:
            raise ValueError("Expected id/source_id/reference/prediction/status fields")
        for key in ("id", "source_id", "reference", "prediction", "status"):
            _text(row[key], key)
        if not row["id"].strip() or not row["source_id"].strip():
            raise ValueError("Nonempty sample and source IDs are required")
        if row["id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["id"])
        if not row["reference"].strip():
            raise ValueError("Corpus reference must contain a non-whitespace target")
        if row["status"] not in {"ok", "empty", "error"}:
            raise ValueError("Unknown prediction status")
        if row["status"] in {"empty", "error"} and row["prediction"] != "":
            raise ValueError("Empty/error prediction must be an empty string")
        if row["status"] == "ok" and row["prediction"] == "":
            raise ValueError("An empty prediction requires empty/error status")
        samples.append({**row, "raw": score_text_v1(row["reference"], row["prediction"]),
                        "normalized": score_text_v1(normalize_text_v1(row["reference"]), normalize_text_v1(row["prediction"]))})
    samples.sort(key=lambda row: row["id"])
    by_source = {}
    for source in sorted({row["source_id"] for row in samples}):
        subset = [row for row in samples if row["source_id"] == source]
        by_source[source] = {"sample_count": len(subset), "status_counts": _status_counts(subset),
                             "raw": _summary([row["raw"] for row in subset]),
                             "normalized": _summary([row["normalized"] for row in subset])}
    return {"policy": deepcopy(COMPARISON_POLICY_V1), "sample_count": len(samples),
            "status_counts": _status_counts(samples),
            "raw": _summary([row["raw"] for row in samples]),
            "normalized": _summary([row["normalized"] for row in samples]),
            "by_source": by_source, "samples": samples}
