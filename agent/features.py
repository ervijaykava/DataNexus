"""
Feature engineering and feature selection.

Both are fitted on the training split only and then replayed on the test split.
That ordering is what makes the reported test scores trustworthy: the test rows
never influence which features exist or which ones survive selection.

Feature engineering here is deliberately restrained. It would be easy to
generate five hundred interaction terms; it would also be impossible to explain
them, and most would be noise. We cap the number of derived features and record
a plain-English reason for every single one.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

MAX_DERIVED_NUMERIC_FEATURES = 12
TOP_NUMERIC_FOR_INTERACTIONS = 4
SKEW_THRESHOLD = 1.5
RARE_CATEGORY_MIN_SHARE = 0.01
MAX_CATEGORIES_KEPT = 20
CORRELATION_REDUNDANCY_THRESHOLD = 0.95
LEAKAGE_CORRELATION_THRESHOLD = 0.98
LEAKAGE_MI_THRESHOLD = 0.95
MAX_FEATURES_AFTER_SELECTION = 60


# --------------------------------------------------------------------------
# Shared helper: a quick numeric view of a mixed dataframe, for scoring only
# --------------------------------------------------------------------------

def _encode_for_scoring(frame):
    """
    Produce an all-numeric copy of a dataframe purely so that correlation and
    mutual information can be computed. This is never used for training.
    """
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            encoded[column] = series.astype("int64", errors="ignore")
        elif pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
        elif pd.api.types.is_bool_dtype(series):
            encoded[column] = series.astype(float)
        else:
            codes, _ = pd.factorize(series.astype(str), use_na_sentinel=True)
            encoded[column] = codes.astype(float)

    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    for column in encoded.columns:
        median = encoded[column].median()
        encoded[column] = encoded[column].fillna(-1.0 if pd.isna(median) else median)
    return encoded


def _mutual_information(features, target, problem_type):
    """Mutual information between each feature and the target, scaled to 0-1."""
    encoded = _encode_for_scoring(features)
    if encoded.shape[1] == 0 or len(encoded) < 10:
        return pd.Series(dtype=float)

    if problem_type == "classification":
        target_values = pd.factorize(pd.Series(target).astype(str))[0]
        scorer = mutual_info_classif
    else:
        target_values = pd.to_numeric(pd.Series(target), errors="coerce").fillna(0).to_numpy()
        scorer = mutual_info_regression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = scorer(encoded, target_values, random_state=42)
        except Exception:
            return pd.Series(dtype=float)

    series = pd.Series(scores, index=encoded.columns)
    maximum = series.max()
    if maximum and maximum > 0:
        series = series / maximum
    return series.sort_values(ascending=False)


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def plan_feature_engineering(features, target, problem_type):
    """
    Look at the training split and decide which derived features to build.

    Returns a "recipe" dictionary that can be replayed on any dataframe with
    the same columns, so training and test data get identical treatment.
    """
    recipe = {
        "datetime_expansions": [],
        "log_transforms": [],
        "ratios": [],
        "differences": [],
        "rare_grouping": {},
        "created": [],
    }

    datetime_columns = [c for c in features.columns
                        if pd.api.types.is_datetime64_any_dtype(features[c])]
    numeric_columns = [c for c in features.columns
                       if pd.api.types.is_numeric_dtype(features[c])]
    categorical_columns = [c for c in features.columns
                           if c not in datetime_columns and c not in numeric_columns]

    # -- Calendar parts from datetime columns -------------------------------
    for column in datetime_columns:
        recipe["datetime_expansions"].append(column)
        for position, part in enumerate(["year", "month", "day", "weekday", "quarter"]):
            # Only the first part carries the full explanation; repeating it five
            # times per date column would bury the rest of the report.
            reason = (f"A model cannot read a raw timestamp, so {column} was split into its "
                      f"calendar parts, which let it learn seasonal and trend patterns."
                      if position == 0 else f"Calendar '{part}' taken from {column}.")
            recipe["created"].append({
                "name": f"{column}_{part}",
                "source": column,
                "type": "datetime part",
                "reason": reason,
            })

    # -- Log transform for heavily right-skewed positive columns ------------
    for column in numeric_columns:
        series = features[column].dropna()
        if len(series) < 20 or series.min() < 0:
            continue
        try:
            skew = float(series.skew())
        except (TypeError, ValueError):
            continue
        if skew > SKEW_THRESHOLD and series.nunique() > 10:
            recipe["log_transforms"].append(column)
            recipe["created"].append({
                "name": f"{column}_log",
                "source": column,
                "type": "log transform",
                "reason": f"{column} is right-skewed (skew {skew:.2f}); a log transform "
                          f"pulls in the long tail so linear models are not dominated by outliers.",
            })

    # -- Ratios and differences between the most informative numeric pairs --
    if len(numeric_columns) >= 2:
        scores = _mutual_information(features[numeric_columns], target, problem_type)
        ranked = [c for c in scores.index if c in numeric_columns][:TOP_NUMERIC_FOR_INTERACTIONS]
        if len(ranked) < 2:
            ranked = numeric_columns[:TOP_NUMERIC_FOR_INTERACTIONS]

        derived_count = 0
        for left, right in itertools.combinations(ranked, 2):
            if derived_count >= MAX_DERIVED_NUMERIC_FEATURES:
                break
            recipe["ratios"].append((left, right))
            recipe["created"].append({
                "name": f"{left}_per_{right}",
                "source": f"{left}, {right}",
                "type": "ratio",
                "reason": f"Ratio of two of the most informative numeric columns. Ratios "
                          f"capture relative size, which raw values on their own cannot.",
            })
            derived_count += 1

            if derived_count >= MAX_DERIVED_NUMERIC_FEATURES:
                break
            recipe["differences"].append((left, right))
            recipe["created"].append({
                "name": f"{left}_minus_{right}",
                "source": f"{left}, {right}",
                "type": "difference",
                "reason": f"Gap between {left} and {right}; differences often matter more "
                          f"than either value alone.",
            })
            derived_count += 1

    # -- Group rare categories ----------------------------------------------
    for column in categorical_columns:
        counts = features[column].astype(str).value_counts(normalize=True)
        if len(counts) <= 10:
            continue
        keep = counts[counts >= RARE_CATEGORY_MIN_SHARE].index.tolist()[:MAX_CATEGORIES_KEPT]
        if len(keep) < len(counts):
            recipe["rare_grouping"][column] = keep
            recipe["created"].append({
                "name": f"{column} (grouped)",
                "source": column,
                "type": "rare category grouping",
                "reason": f"{len(counts) - len(keep)} rare category value(s) in {column} were "
                          f"merged into 'Other'. Encoding each of them would create many "
                          f"near-empty columns that only add noise.",
            })

    return recipe


def apply_feature_engineering(features, recipe):
    """Replay a recipe on a dataframe. Safe to call on train and test alike."""
    engineered = features.copy()

    for column in recipe["datetime_expansions"]:
        if column not in engineered.columns:
            continue
        series = pd.to_datetime(engineered[column], errors="coerce")
        engineered[f"{column}_year"] = series.dt.year
        engineered[f"{column}_month"] = series.dt.month
        engineered[f"{column}_day"] = series.dt.day
        engineered[f"{column}_weekday"] = series.dt.weekday
        engineered[f"{column}_quarter"] = series.dt.quarter
        engineered = engineered.drop(columns=[column])

    for column in recipe["log_transforms"]:
        if column in engineered.columns:
            engineered[f"{column}_log"] = np.log1p(engineered[column].clip(lower=0))

    for left, right in recipe["ratios"]:
        if left in engineered.columns and right in engineered.columns:
            denominator = engineered[right].replace(0, np.nan)
            engineered[f"{left}_per_{right}"] = engineered[left] / denominator

    for left, right in recipe["differences"]:
        if left in engineered.columns and right in engineered.columns:
            engineered[f"{left}_minus_{right}"] = engineered[left] - engineered[right]

    for column, keep in recipe["rare_grouping"].items():
        if column in engineered.columns:
            values = engineered[column].astype(str)
            engineered[column] = values.where(values.isin(keep), other="Other")

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered


# --------------------------------------------------------------------------
# Leakage detection
# --------------------------------------------------------------------------

def detect_leakage(features, target, problem_type):
    """
    Find columns that predict the target too well to be honest.

    This runs on the raw columns *before* feature engineering, because a leaky
    column would otherwise be baked into every ratio and difference derived
    from it and survive selection in disguise.

    The classification rule deliberately ignores near-unique columns: a column
    with one distinct value per row trivially "separates the classes" without
    that meaning anything at all.
    """
    findings = []
    row_count = max(len(features), 1)
    label_like_limit = max(20, int(0.05 * row_count))

    if problem_type == "regression":
        numeric_target = pd.to_numeric(pd.Series(target), errors="coerce")
        for column in features.columns:
            if not pd.api.types.is_numeric_dtype(features[column]):
                continue
            try:
                correlation = abs(features[column].corr(numeric_target))
            except Exception:
                continue
            if pd.notna(correlation) and correlation >= LEAKAGE_CORRELATION_THRESHOLD:
                findings.append({
                    "feature": column,
                    "evidence": f"its correlation with the target is {correlation:.3f}",
                    "reason": "A feature this close to the target is usually a copy of it, or "
                              "information that would only be known after the outcome.",
                })
        return findings

    target_strings = pd.Series(target).astype(str).reset_index(drop=True)
    for column in features.columns:
        values = features[column]
        distinct = values.nunique(dropna=True)
        # Only categorical-sized columns can meaningfully "determine" a class.
        if distinct <= 2 or distinct > label_like_limit:
            continue
        grouped = pd.DataFrame({
            "f": values.astype(str).reset_index(drop=True),
            "y": target_strings,
        })
        purity = grouped.groupby("f")["y"].nunique()
        if (purity <= 1).all():
            findings.append({
                "feature": column,
                "evidence": f"each of its {distinct} values maps to exactly one target class",
                "reason": "This feature determines the answer perfectly, which almost always "
                          "means it was derived from the target.",
            })
    return findings


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------

def select_features(features, target, problem_type):
    """
    Decide which features to keep. Returns (kept, removed, scores, leakage).

    `removed` and `leakage` both carry human-readable reasons, because "which
    features did you drop and why" is the first question anyone asks of an
    automated pipeline.
    """
    removed = []
    leakage = []
    candidates = list(features.columns)

    def drop(column, reason):
        if column in candidates:
            candidates.remove(column)
            removed.append({"feature": column, "reason": reason})

    # -- 1. Constant / near-constant ----------------------------------------
    for column in list(candidates):
        unique = features[column].nunique(dropna=True)
        if unique <= 1:
            drop(column, "Only one distinct value, so it cannot explain any variation.")
        elif unique == 2 and features[column].value_counts(normalize=True).iloc[0] > 0.999:
            drop(column, "Over 99.9% of rows share the same value - effectively constant.")

    if not candidates:
        return [], removed, pd.Series(dtype=float), leakage

    scores = _mutual_information(features[candidates], target, problem_type)

    # -- 2. Suspiciously perfect predictors (likely leakage) -----------------
    for item in detect_leakage(features[candidates], target, problem_type):
        leakage.append(item)
        drop(item["feature"], f"Removed as suspected target leakage - {item['evidence']}.")

    # -- 3. Redundant, highly correlated numeric pairs ----------------------
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(features[c])]
    if len(numeric_candidates) > 1:
        correlation_matrix = features[numeric_candidates].corr().abs()
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        for left in upper.columns:
            for right in upper.index:
                value = upper.loc[right, left]
                if pd.isna(value) or value <= CORRELATION_REDUNDANCY_THRESHOLD:
                    continue
                if left not in candidates or right not in candidates:
                    continue
                weaker = left if scores.get(left, 0) < scores.get(right, 0) else right
                stronger = right if weaker == left else left
                drop(weaker,
                     f"Correlated {value:.3f} with {stronger}, which carries the same "
                     f"information and scores higher against the target.")

    # -- 4. Cap the total number of features --------------------------------
    if len(candidates) > MAX_FEATURES_AFTER_SELECTION:
        ranked = [c for c in scores.index if c in candidates]
        ranked += [c for c in candidates if c not in ranked]
        for column in ranked[MAX_FEATURES_AFTER_SELECTION:]:
            drop(column,
                 f"Ranked outside the top {MAX_FEATURES_AFTER_SELECTION} features by mutual "
                 f"information with the target; kept out to control overfitting and training time.")

    # -- 5. Never return an empty feature set -------------------------------
    if not candidates:
        rescued = [c for c in scores.index][:5] or list(features.columns)[:5]
        candidates = rescued
        removed = [r for r in removed if r["feature"] not in candidates]

    kept_scores = scores[[c for c in scores.index if c in candidates]] if len(scores) else scores
    return candidates, removed, kept_scores, leakage
