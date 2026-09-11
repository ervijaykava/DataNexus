"""
Step 1 of the agent: understand the dataset.

Everything here runs locally on the full dataset. The point is to know the data
*before* we ask Gemini anything - so that the AI receives a compact, factual
profile instead of a raw CSV dump.
"""

import re
import warnings

import numpy as np
import pandas as pd

# Column names that very often mean "this is the thing to predict".
TARGET_NAME_HINTS = [
    "target", "label", "class", "outcome", "result", "y",
    "churn", "churned", "attrition", "exited", "left",
    "price", "sales", "revenue", "profit", "cost", "amount", "salary", "income",
    "survived", "default", "fraud", "is_fraud", "click", "converted", "conversion",
    "rating", "score", "risk", "status", "approved", "success", "response",
]

# Column names that almost always mean "this is a row identifier".
ID_NAME_HINTS = ["id", "uuid", "guid", "index", "key", "code", "no", "number", "serial"]

MAX_ID_LIKE_UNIQUE_RATIO = 0.95
HIGH_CARDINALITY_THRESHOLD = 50
HIGH_MISSING_THRESHOLD = 0.40


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _clean_name(name):
    """Lowercase a column name and reduce it to words, for hint matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _looks_like_datetime(series, threshold=0.80):
    """
    Return True if most non-null values in an object column parse as dates.

    We test a sample rather than the whole column so that this stays fast on
    large files, and we swallow parsing warnings because "it did not parse" is
    a perfectly normal answer here.
    """
    sample = series.dropna().astype(str).head(200)
    if len(sample) < 5:
        return False

    # A column of plain integers ("2019", "100234") should not be treated as a
    # date just because pandas can coerce it.
    if sample.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.5:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
            except Exception:
                return False
    return parsed.notna().mean() >= threshold


def _looks_like_number(series, threshold=0.90):
    """
    Return True if an object column is really numbers wearing a text costume,
    e.g. "1,234", "$45.00", "88%".
    """
    sample = series.dropna().astype(str).head(500)
    if len(sample) < 5:
        return False
    stripped = sample.str.replace(r"[,$₹€£%\s]", "", regex=True)
    converted = pd.to_numeric(stripped, errors="coerce")
    return converted.notna().mean() >= threshold


def _safe_float(value):
    """Convert numpy scalars to plain floats, turning NaN/inf into None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, 6)


def classify_column(series):
    """Decide what kind of column this is. Returns one of the KIND_* strings."""
    non_null = series.dropna()

    if len(non_null) == 0:
        return "empty"
    if series.nunique(dropna=True) <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # An integer column with only two values is really a boolean flag.
        if series.nunique(dropna=True) == 2:
            return "binary"
        return "numeric"

    if _looks_like_datetime(series):
        return "datetime"
    if _looks_like_number(series):
        return "numeric"

    unique_ratio = series.nunique(dropna=True) / max(len(non_null), 1)
    average_length = non_null.astype(str).str.len().mean()
    if unique_ratio > 0.9 and average_length > 25:
        return "text"
    return "categorical"


# --------------------------------------------------------------------------
# Column-level profile
# --------------------------------------------------------------------------

def profile_column(series, n_rows):
    kind = classify_column(series)
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))

    info = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "kind": kind,
        "missing": missing,
        "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
        "unique": unique,
        "unique_pct": round(100 * unique / n_rows, 2) if n_rows else 0.0,
        "sample_values": [str(v) for v in series.dropna().unique()[:5]],
    }

    if kind in ("numeric", "binary") and pd.api.types.is_numeric_dtype(series):
        # Infinite values would poison mean/std/skew, so they are excluded from
        # the statistics here and reported separately as a data quality issue.
        finite = series.replace([np.inf, -np.inf], np.nan)
        described = finite.describe()
        info.update({
            "min": _safe_float(described.get("min")),
            "max": _safe_float(described.get("max")),
            "mean": _safe_float(described.get("mean")),
            "std": _safe_float(described.get("std")),
            "median": _safe_float(finite.median()),
            "skew": _safe_float(finite.skew()),
            "zeros": int((series == 0).sum()),
            "negatives": int((series < 0).sum()),
            "infinite": int(np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).sum()),
        })
    elif kind in ("categorical", "binary", "boolean"):
        counts = series.value_counts(dropna=True).head(5)
        info["top_values"] = [
            {"value": str(index), "count": int(count)} for index, count in counts.items()
        ]

    return info


# --------------------------------------------------------------------------
# Target candidate scoring
# --------------------------------------------------------------------------

def score_target_candidates(column_profiles, n_rows):
    """
    Rank columns by how plausible they are as a prediction target.

    This is a local, explainable heuristic. Gemini gets a vote too, but this
    is what we validate its answer against.
    """
    candidates = []

    for info in column_profiles:
        name = _clean_name(info["name"])
        kind = info["kind"]
        score = 0.0
        reasons = []

        if kind in ("empty", "constant", "text", "datetime"):
            continue

        # Name signals.
        for hint in TARGET_NAME_HINTS:
            if name == hint:
                score += 45
                reasons.append(f"column name is exactly '{hint}'")
                break
            if hint in name.split("_"):
                score += 30
                reasons.append(f"column name contains '{hint}'")
                break

        # Position: targets are very often the last column.
        if info.get("is_last_column"):
            score += 10
            reasons.append("last column in the file")

        # Shape signals.
        if kind == "binary":
            score += 25
            reasons.append("two distinct values (binary outcome)")
        elif kind == "categorical" and 2 <= info["unique"] <= 20:
            score += 15
            reasons.append(f"{info['unique']} distinct classes")
        elif kind == "numeric":
            score += 10
            reasons.append("continuous numeric column")

        # Penalties.
        if info["unique_pct"] > 95 and kind != "numeric":
            score -= 40
            reasons.append("nearly every value is unique (looks like an identifier)")
        if any(hint == name or name.endswith("_" + hint) for hint in ID_NAME_HINTS):
            score -= 45
            reasons.append("name looks like an identifier")
        if info["missing_pct"] > 30:
            score -= 20
            reasons.append(f"{info['missing_pct']}% missing")

        if score <= 0:
            continue

        if kind in ("binary", "boolean"):
            likely_task = "classification"
        elif kind == "categorical":
            likely_task = "classification"
        else:
            likely_task = "regression"

        candidates.append({
            "column": info["name"],
            "score": round(score, 1),
            "likely_task": likely_task,
            "reason": "; ".join(reasons) if reasons else "plausible shape",
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:8]


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------

def assess_quality(profile):
    """Turn the raw profile into a 0-100 score plus a list of readable issues."""
    issues = []
    score = 100

    if profile["missing_pct"] > 0:
        severity = "high" if profile["missing_pct"] > 20 else (
            "medium" if profile["missing_pct"] > 5 else "low")
        penalty = min(25, profile["missing_pct"])
        score -= penalty
        issues.append({
            "severity": severity,
            "message": f"{profile['missing_pct']}% of all cells are missing "
                       f"({profile['total_missing_cells']:,} cells).",
        })

    if profile["duplicate_rows"] > 0:
        score -= min(15, profile["duplicate_pct"])
        issues.append({
            "severity": "medium" if profile["duplicate_pct"] > 5 else "low",
            "message": f"{profile['duplicate_rows']:,} duplicate rows "
                       f"({profile['duplicate_pct']}% of the dataset).",
        })

    if profile["constant_columns"]:
        score -= 5
        issues.append({
            "severity": "low",
            "message": f"{len(profile['constant_columns'])} column(s) hold a single "
                       f"repeated value and carry no information: "
                       f"{', '.join(profile['constant_columns'][:5])}.",
        })

    if profile["id_like_columns"]:
        issues.append({
            "severity": "low",
            "message": f"{len(profile['id_like_columns'])} identifier-like column(s) "
                       f"detected and excluded from modelling: "
                       f"{', '.join(profile['id_like_columns'][:5])}.",
        })

    if profile["high_missing_columns"]:
        score -= 10
        issues.append({
            "severity": "high",
            "message": f"Column(s) with more than 40% missing values: "
                       f"{', '.join(profile['high_missing_columns'][:5])}.",
        })

    if profile["high_cardinality_columns"]:
        issues.append({
            "severity": "medium",
            "message": f"High-cardinality categorical column(s) that will be grouped "
                       f"before encoding: {', '.join(profile['high_cardinality_columns'][:5])}.",
        })

    if profile["n_rows"] < 100:
        score -= 20
        issues.append({
            "severity": "high",
            "message": f"Only {profile['n_rows']} rows. Results from such a small "
                       f"sample will not be statistically reliable.",
        })

    if not issues:
        issues.append({"severity": "none", "message": "No significant data quality problems detected."})

    return max(0, int(round(score))), issues


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def profile_dataframe(dataframe, sample_rows=8):
    """Build the complete dataset profile used by the rest of the agent."""
    n_rows, n_columns = dataframe.shape

    column_profiles = []
    for position, name in enumerate(dataframe.columns):
        info = profile_column(dataframe[name], n_rows)
        info["is_last_column"] = (position == n_columns - 1)
        info["position"] = position
        column_profiles.append(info)

    by_kind = {}
    for info in column_profiles:
        by_kind.setdefault(info["kind"], []).append(info["name"])

    # Identifier detection: name hint, or almost-unique values.
    id_like = []
    for info in column_profiles:
        name = _clean_name(info["name"])
        name_says_id = any(name == hint or name.endswith("_" + hint) or name.startswith(hint + "_")
                           for hint in ID_NAME_HINTS)
        nearly_unique = info["unique_pct"] >= MAX_ID_LIKE_UNIQUE_RATIO * 100
        # Free-text columns are handled separately so they get an accurate
        # explanation rather than being labelled identifiers.
        if (name_says_id and nearly_unique) or (nearly_unique and info["kind"] == "categorical"):
            id_like.append(info["name"])
        elif name_says_id and info["kind"] == "numeric" and nearly_unique:
            id_like.append(info["name"])

    # Identifiers are dropped outright, so listing them again as "high
    # cardinality columns that will be grouped" would be misleading.
    high_cardinality = [
        info["name"] for info in column_profiles
        if info["kind"] == "categorical"
        and info["unique"] > HIGH_CARDINALITY_THRESHOLD
        and info["name"] not in id_like
    ]
    high_missing = [
        info["name"] for info in column_profiles
        if info["missing_pct"] > HIGH_MISSING_THRESHOLD * 100
    ]

    duplicate_rows = int(dataframe.duplicated().sum())
    total_cells = n_rows * n_columns
    total_missing = int(dataframe.isna().sum().sum())

    sample = dataframe.head(sample_rows).copy()
    sample_records = [
        {str(k): ("" if pd.isna(v) else str(v)) for k, v in record.items()}
        for record in sample.to_dict(orient="records")
    ]

    profile = {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "memory_mb": round(dataframe.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(100 * duplicate_rows / n_rows, 2) if n_rows else 0.0,
        "total_missing_cells": total_missing,
        "missing_pct": round(100 * total_missing / total_cells, 2) if total_cells else 0.0,
        "columns": column_profiles,
        "column_names": [str(c) for c in dataframe.columns],
        "numeric_columns": by_kind.get("numeric", []) + by_kind.get("binary", []),
        "categorical_columns": by_kind.get("categorical", []) + by_kind.get("boolean", []),
        "datetime_columns": by_kind.get("datetime", []),
        "text_columns": by_kind.get("text", []),
        "constant_columns": by_kind.get("constant", []) + by_kind.get("empty", []),
        "id_like_columns": id_like,
        "high_cardinality_columns": high_cardinality,
        "high_missing_columns": high_missing,
        "sample_rows": sample_records,
    }

    profile["target_candidates"] = score_target_candidates(column_profiles, n_rows)
    profile["quality_score"], profile["quality_issues"] = assess_quality(profile)

    return profile
