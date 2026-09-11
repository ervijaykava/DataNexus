"""
Step 2 of the agent: clean the dataset.

Two rules guide everything in this module:

1. Never throw away information we cannot justify. Every drop is recorded with
   a reason that ends up in the UI and in the final report.
2. Do structural cleaning here, but leave *missing value imputation* to the
   modelling pipeline. Imputing before the train/test split would let the test
   set influence the training statistics - a classic and invisible source of
   data leakage. The plan is decided here and reported; the fitting happens on
   the training fold only.
"""

import re
import warnings

import numpy as np
import pandas as pd

from agent import profiling

# Strings that mean "missing" but arrive as ordinary text.
NULL_TOKENS = {
    "", " ", "na", "n/a", "n.a.", "nan", "null", "none", "nil",
    "-", "--", "?", "unknown", "missing", "not available", "#n/a",
}

# A column this empty cannot be repaired, so it is dropped rather than imputed.
DROP_IF_MISSING_ABOVE = 60.0

MISSING_STRATEGY = {
    "numeric": "median",
    "categorical": "most_frequent",
}


def _is_text_column(series):
    """
    True for text columns whichever dtype pandas chose.

    pandas 2.x stores text as `object` while pandas 3.x may use the dedicated
    `str` dtype, so checking `== "object"` alone silently skips every text
    column on newer pandas. This keeps the cleaning rules working on both.
    """
    return (pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)) and not pd.api.types.is_numeric_dtype(series)


def _snapshot(dataframe):
    """The small set of numbers we show in the before/after comparison."""
    rows, columns = dataframe.shape
    total_cells = rows * columns
    missing = int(dataframe.isna().sum().sum())
    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_cells": missing,
        "missing_pct": round(100 * missing / total_cells, 2) if total_cells else 0.0,
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def _to_number(series):
    """Parse '1,234', '$45.00', '88%' into real numbers."""
    stripped = series.astype(str).str.replace(r"[,$₹€£%\s]", "", regex=True)
    return pd.to_numeric(stripped, errors="coerce")


def clean_dataframe(dataframe, profile, protect=None):
    """
    Clean a dataframe and return (cleaned_dataframe, cleaning_report).

    `protect` is a list of column names that must survive cleaning no matter
    what - in practice, the target column.
    """
    protect = set(protect or [])
    working = dataframe.copy()
    steps = []
    dropped_columns = []
    before = _snapshot(working)

    def record(action, detail, columns=None):
        steps.append({
            "action": action,
            "detail": detail,
            "columns": columns or [],
        })

    # -- 1. Trim stray whitespace -------------------------------------------
    object_columns = [c for c in working.columns if _is_text_column(working[c])]
    trimmed = []
    for column in object_columns:
        original = working[column]
        stripped = original.astype(str).str.strip()
        # astype(str) turns NaN into the literal "nan", so restore the nulls.
        stripped = stripped.where(original.notna(), other=np.nan)
        if not stripped.equals(original):
            working[column] = stripped
            trimmed.append(column)
    if trimmed:
        record("Trimmed whitespace",
               f"Removed leading/trailing spaces in {len(trimmed)} text column(s).",
               trimmed)

    # -- 2. Turn placeholder text into real nulls ---------------------------
    tokenised = []
    for column in object_columns:
        lowered = working[column].astype(str).str.strip().str.lower()
        mask = lowered.isin(NULL_TOKENS) & working[column].notna()
        if mask.any():
            working.loc[mask, column] = np.nan
            tokenised.append(column)
    if tokenised:
        record("Normalised placeholder values",
               f"Converted text placeholders such as 'N/A', '?', '-' and 'unknown' "
               f"into proper missing values in {len(tokenised)} column(s).",
               tokenised)

    # -- 3. Fix inconsistent casing (only when it is clearly the same label) --
    recased = []
    for column in object_columns:
        series = working[column].dropna()
        if series.empty or series.nunique() > 200:
            continue
        lowered = series.astype(str).str.lower()
        # If lowercasing merges categories, the column had case variants of the
        # same label ("Male"/"male"). That is safe to normalise.
        if lowered.nunique() < series.nunique():
            working[column] = working[column].astype(str).str.lower().where(
                working[column].notna(), other=np.nan)
            recased.append(column)
    if recased:
        record("Unified category casing",
               f"Merged case-variant categories (for example 'Male' and 'male') "
               f"in {len(recased)} column(s).",
               recased)

    # -- 4. Recover numbers stored as text ----------------------------------
    converted_numeric = []
    for column in list(working.columns):
        if not _is_text_column(working[column]):
            continue
        if profiling._looks_like_number(working[column]):
            converted = _to_number(working[column])
            # Only accept the conversion if it does not create new nulls.
            new_nulls = int(converted.isna().sum() - working[column].isna().sum())
            if new_nulls <= 0.02 * len(working):
                working[column] = converted
                converted_numeric.append(column)
    if converted_numeric:
        record("Repaired numeric columns",
               f"Parsed {len(converted_numeric)} column(s) that were stored as text "
               f"with symbols such as currency signs, thousands separators or "
               f"percent signs into true numeric columns.",
               converted_numeric)

    # -- 5. Parse date columns ----------------------------------------------
    converted_dates = []
    for column in list(working.columns):
        if not _is_text_column(working[column]):
            continue
        if profiling._looks_like_datetime(working[column]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    parsed = pd.to_datetime(working[column], errors="coerce", format="mixed")
                except (ValueError, TypeError):
                    try:
                        parsed = pd.to_datetime(working[column], errors="coerce")
                    except Exception:
                        continue
            if parsed.notna().mean() >= 0.80:
                working[column] = parsed
                converted_dates.append(column)
    if converted_dates:
        record("Parsed date columns",
               f"Converted {len(converted_dates)} text column(s) into real datetime "
               f"columns so that calendar features can be derived from them.",
               converted_dates)

    # -- 6. Replace infinities ----------------------------------------------
    numeric_columns = working.select_dtypes(include=[np.number]).columns
    infinite_columns = []
    for column in numeric_columns:
        values = working[column].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(values).any():
            working[column] = working[column].replace([np.inf, -np.inf], np.nan)
            infinite_columns.append(column)
    if infinite_columns:
        record("Replaced infinite values",
               f"Infinite values in {len(infinite_columns)} column(s) were replaced "
               f"with missing values so they can be imputed rather than crash the models.",
               infinite_columns)

    # -- 7. Remove duplicate rows -------------------------------------------
    duplicate_count = int(working.duplicated().sum())
    if duplicate_count:
        working = working.drop_duplicates().reset_index(drop=True)
        record("Removed duplicate rows",
               f"Dropped {duplicate_count:,} exactly duplicated row(s) "
               f"({round(100 * duplicate_count / max(before['rows'], 1), 2)}% of the dataset).")

    # -- 8. Drop columns that cannot help -----------------------------------
    def drop_columns(columns, reason, action):
        nonlocal working
        columns = [c for c in columns if c in working.columns and c not in protect]
        if not columns:
            return
        working = working.drop(columns=columns)
        dropped_columns.extend([{"column": c, "reason": reason} for c in columns])
        record(action, reason, columns)

    constant_now = [c for c in working.columns if working[c].nunique(dropna=True) <= 1]
    drop_columns(constant_now,
                 "Holds a single repeated value, so it cannot explain any variation in the target.",
                 "Dropped constant columns")

    fresh_profile = profiling.profile_dataframe(working, sample_rows=3)

    drop_columns(fresh_profile["id_like_columns"],
                 "Looks like a row identifier. Identifiers are unique per row and would "
                 "let a model memorise rows instead of learning a pattern.",
                 "Dropped identifier columns")

    drop_columns(fresh_profile["text_columns"],
                 "Free-text column with almost entirely unique long values. Modelling it "
                 "properly needs natural-language processing, which is outside this agent's scope.",
                 "Dropped free-text columns")

    too_empty = [
        info["name"] for info in fresh_profile["columns"]
        if info["missing_pct"] > DROP_IF_MISSING_ABOVE
    ]
    drop_columns(too_empty,
                 f"More than {DROP_IF_MISSING_ABOVE:.0f}% of the values are missing, so "
                 f"imputing it would invent more data than it preserves.",
                 "Dropped near-empty columns")

    after = _snapshot(working)

    report = {
        "steps": steps,
        "before": before,
        "after": after,
        "dropped_columns": dropped_columns,
        "rows_removed": before["rows"] - after["rows"],
        "columns_removed": before["columns"] - after["columns"],
        "missing_strategy": MISSING_STRATEGY,
        "missing_note": (
            "Remaining missing values are imputed inside the modelling pipeline "
            "(numeric columns with the median, categorical columns with the most "
            "frequent value). The imputers are fitted on the training split only, "
            "so no information from the test set leaks into training."
        ),
    }
    return working, report


def drop_rows_missing_target(dataframe, target_column):
    """
    Rows without a target value cannot be used for supervised learning and must
    not be imputed - inventing a label would be inventing the answer.
    """
    missing = int(dataframe[target_column].isna().sum())
    if missing == 0:
        return dataframe, 0
    cleaned = dataframe[dataframe[target_column].notna()].reset_index(drop=True)
    return cleaned, missing


def impute_for_exploration(dataframe):
    """
    Simple direct imputation used only for exploratory (non-modelling) runs,
    where there is no train/test split and therefore no leakage risk.
    """
    filled = dataframe.copy()
    for column in filled.columns:
        if filled[column].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(filled[column]):
            filled[column] = filled[column].fillna(filled[column].median())
        else:
            mode = filled[column].mode(dropna=True)
            filled[column] = filled[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return filled
