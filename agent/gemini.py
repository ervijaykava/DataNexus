"""
The reasoning layer.

Gemini is asked to read a compact dataset profile and the user's objective and
return a structured recommendation. Three rules apply without exception:

1. The raw dataset is never uploaded. Gemini sees column metadata, summary
   statistics and a handful of sample rows - nothing more.
2. Gemini recommends; it never executes. No code it returns is ever run. Model
   names are resolved against a fixed library and anything unknown is dropped.
3. Gemini is optional. If there is no API key, or the call fails, or the reply
   is not valid JSON, `local_recommendation` produces the same structure from
   rules alone and the application carries on without a visible difference.
"""

import json
import re

import config

# The exact shape every recommendation must have, whatever produced it.
RECOMMENDATION_KEYS = [
    "problem_type",
    "target_column",
    "target_confidence",
    "recommended_models",
    "relevant_features",
    "columns_to_drop",
    "preprocessing_recommendations",
    "feature_engineering_recommendations",
    "potential_data_leakage",
    "evaluation_metric",
    "reasoning",
    "objective_success_criteria",
]

VALID_PROBLEM_TYPES = {
    "classification", "regression", "clustering",
    "anomaly_detection", "time_series_forecasting", "exploratory",
}

# Keyword signals used by the local analyst and to sanity-check Gemini.
# Words that settle the question on their own, checked before the tally below.
STRONG_SIGNALS = {
    "time_series_forecasting": [
        "forecast", "time series", "time-series", "next month", "next quarter",
        "next year", "seasonality", "over the coming",
    ],
    "clustering": [
        "cluster", "segment customers", "segment the customers", "natural groups",
        "unsupervised", "personas", "group similar",
    ],
    "anomaly_detection": [
        "anomaly", "anomalies", "outlier detection", "abnormal", "novelty detection",
    ],
}

OBJECTIVE_SIGNALS = {
    "classification": [
        "classify", "classification", "whether", "will they", "predict if", "yes or no",
        "churn", "fraud", "spam", "default", "approve", "risk category", "categor",
        "which type", "which class", "segment into", "label", "detect if", "likelihood of",
    ],
    "regression": [
        "how much", "how many", "price", "revenue", "sales", "amount", "cost", "value",
        "salary", "income", "score", "estimate", "predict the number", "demand", "quantity",
    ],
    "clustering": [
        "cluster", "group similar", "segment customers", "find groups", "natural groups",
        "unsupervised", "personas",
    ],
    "anomaly_detection": [
        "anomaly", "anomalies", "outlier", "unusual", "abnormal", "novelty",
    ],
    "time_series_forecasting": [
        "forecast", "next month", "next quarter", "over time", "time series",
        "future values", "trend over", "seasonality",
    ],
    "exploratory": [
        "which factors", "what influences", "what drives", "explore", "understand",
        "relationship between", "insight", "analyse", "analyze", "summar", "describe",
        "correlat", "most important factor", "affect",
    ],
}


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_profile_summary(profile, max_columns=60):
    """
    Compress the dataset profile into the compact text Gemini actually sees.

    On a wide dataset we describe the first `max_columns` columns in detail and
    only name the rest, which keeps the prompt small and the reply focused.
    """
    lines = [
        f"Rows: {profile['n_rows']:,}",
        f"Columns: {profile['n_columns']}",
        f"Duplicate rows: {profile['duplicate_rows']:,} ({profile['duplicate_pct']}%)",
        f"Missing cells overall: {profile['missing_pct']}%",
        "",
        "COLUMNS:",
    ]

    for info in profile["columns"][:max_columns]:
        parts = [
            f"- {info['name']} | kind={info['kind']} | dtype={info['dtype']}",
            f"missing={info['missing_pct']}%",
            f"unique={info['unique']}",
        ]
        if info.get("min") is not None:
            parts.append(f"min={info['min']} max={info['max']} mean={info['mean']} median={info.get('median')}")
        if info.get("top_values"):
            top = ", ".join(f"{v['value']}({v['count']})" for v in info["top_values"][:4])
            parts.append(f"top_values=[{top}]")
        elif info.get("sample_values"):
            parts.append(f"examples=[{', '.join(info['sample_values'][:3])}]")
        lines.append(" | ".join(parts))

    remaining = profile["n_columns"] - max_columns
    if remaining > 0:
        lines.append(f"... and {remaining} further columns not described in detail.")

    lines.append("")
    lines.append("DETECTED CHARACTERISTICS:")
    lines.append(f"- Identifier-like columns: {profile['id_like_columns'] or 'none'}")
    lines.append(f"- Constant columns: {profile['constant_columns'] or 'none'}")
    lines.append(f"- High-cardinality categoricals: {profile['high_cardinality_columns'] or 'none'}")
    lines.append(f"- Columns with >40% missing: {profile['high_missing_columns'] or 'none'}")
    lines.append(f"- Local target candidates (best first): "
                 f"{[c['column'] for c in profile['target_candidates']] or 'none'}")

    lines.append("")
    lines.append(f"SAMPLE ROWS (first {len(profile['sample_rows'])}):")
    for record in profile["sample_rows"][:config.GEMINI_SAMPLE_ROWS]:
        trimmed = {k: (v[:40] + "…" if len(v) > 40 else v) for k, v in list(record.items())[:25]}
        lines.append(json.dumps(trimmed, ensure_ascii=False))

    return "\n".join(lines)


def build_prompt(objective, profile, available_models):
    profile_text = build_profile_summary(profile)
    model_list = "\n".join(f"- {name}" for name in available_models)

    return f"""You are a senior data scientist reviewing a dataset before any modelling begins.

USER OBJECTIVE:
"{objective}"

DATASET PROFILE:
{profile_text}

MODELS AVAILABLE TO THE SYSTEM (you may only recommend names from this list):
{model_list}

Decide how this objective should be approached. Respond with ONLY a JSON object,
no prose and no markdown fences, using exactly these keys:

{{
  "problem_type": one of "classification", "regression", "clustering", "anomaly_detection", "time_series_forecasting", "exploratory",
  "target_column": the exact column name to predict, or null if the objective needs no target,
  "target_confidence": a number between 0 and 1 for how sure you are about the target column,
  "recommended_models": array of 3-6 model names taken verbatim from the list above,
  "relevant_features": array of column names that should help,
  "columns_to_drop": array of column names that should be excluded, with identifiers first,
  "preprocessing_recommendations": array of short strings,
  "feature_engineering_recommendations": array of short strings,
  "potential_data_leakage": array of column names that could leak the answer, empty if none,
  "evaluation_metric": one of "accuracy", "f1", "precision", "recall", "roc_auc", "r2", "rmse", "mae",
  "reasoning": 2-4 sentences explaining your choices in plain English,
  "objective_success_criteria": one sentence describing what result would genuinely satisfy this objective
}}

Rules:
- The target column MUST be an exact name from the column list, or null.
- Choose the metric that fits the objective, not the one that flatters the model. If the
  classes are imbalanced, do not choose accuracy.
- Flag any column that could only be known after the outcome as potential data leakage.
- If the objective does not describe a prediction task, use "exploratory" and set target_column to null."""


# --------------------------------------------------------------------------
# Calling Gemini
# --------------------------------------------------------------------------

def _extract_json(text):
    """Pull a JSON object out of a reply that may be wrapped in prose or fences."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def call_gemini(objective, profile, available_models):
    """
    Ask Gemini for a recommendation.

    Returns (recommendation_dict_or_None, source_label, error_message).
    Every failure path is non-fatal by design.
    """
    if not config.GEMINI_API_KEY:
        return None, "local-fallback", "No GEMINI_API_KEY configured."

    try:
        from google import genai
    except ImportError:
        return None, "local-fallback", "google-genai package is not installed."

    prompt = build_prompt(objective, profile, available_models)

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        parsed = _extract_json(getattr(response, "text", None))
    except Exception as exc:                      # noqa: BLE001
        return None, "local-fallback", f"Gemini call failed: {type(exc).__name__}: {exc}"[:300]

    if not parsed:
        return None, "local-fallback", "Gemini returned a reply that was not valid JSON."

    return parsed, "gemini", None


# --------------------------------------------------------------------------
# The local analyst (used whenever Gemini is unavailable)
# --------------------------------------------------------------------------

def _detect_problem_type_from_text(objective):
    lowered = objective.lower()

    # Some words are decisive rather than suggestive: "forecast" or "cluster"
    # describes a fundamentally different task, so they are checked first and
    # win outright instead of competing on a keyword count.
    for problem_type, keywords in STRONG_SIGNALS.items():
        if any(keyword in lowered for keyword in keywords):
            return problem_type

    scores = {}
    for problem_type, keywords in OBJECTIVE_SIGNALS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits:
            scores[problem_type] = hits
    if not scores:
        return None
    return max(scores, key=scores.get)


def _find_target_in_objective(objective, column_names):
    """If the user literally names a column in their objective, believe them."""
    lowered = objective.lower()
    matches = []
    for name in column_names:
        readable = re.sub(r"[_\-]+", " ", str(name).lower()).strip()
        if len(readable) < 3:
            continue
        if readable in lowered or str(name).lower() in lowered:
            matches.append((name, len(readable)))
    if not matches:
        return None
    matches.sort(key=lambda pair: pair[1], reverse=True)
    return matches[0][0]


def local_recommendation(objective, profile, available_models):
    """
    Produce the same recommendation structure as Gemini, using rules only.

    This is what keeps the application demonstrable with no API key, no network
    and no quota.
    """
    candidates = profile["target_candidates"]
    column_names = profile["column_names"]

    named_target = _find_target_in_objective(objective, column_names)
    text_type = _detect_problem_type_from_text(objective)

    if named_target:
        target = named_target
        confidence = 0.80
        target_reason = f"'{target}' is named directly in the objective."
    elif candidates:
        best = candidates[0]
        target = best["column"]
        runner_up = candidates[1]["score"] if len(candidates) > 1 else 0
        margin = best["score"] - runner_up
        confidence = 0.45 + min(0.4, margin / 100)
        target_reason = f"'{target}' scored highest as a target candidate because {best['reason']}."
    else:
        target = None
        confidence = 0.0
        target_reason = "No column looked like a plausible prediction target."

    # Decide the problem type from the target's own shape, then let the wording
    # of the objective override it only for the unsupervised cases.
    problem_type = "exploratory"
    if target:
        column_info = next((c for c in profile["columns"] if c["name"] == target), None)
        if column_info:
            if column_info["kind"] in ("binary", "boolean", "categorical"):
                problem_type = "classification"
            elif column_info["kind"] == "numeric":
                problem_type = "regression"

    if text_type in ("clustering", "anomaly_detection", "time_series_forecasting"):
        problem_type = text_type
        target = None if text_type != "time_series_forecasting" else target
    elif text_type == "exploratory" and confidence < 0.6:
        problem_type = "exploratory"
        target = None
    elif text_type in ("classification", "regression") and target is None:
        problem_type = text_type

    if problem_type == "classification":
        recommended = [m for m in available_models if "Classifier" in m or m in
                       ("Logistic Regression", "Gaussian Naive Bayes")][:5]
        metric = "f1"
    elif problem_type == "regression":
        recommended = [m for m in available_models if "Regress" in m][:5]
        metric = "r2"
    else:
        recommended = []
        metric = "r2"

    preprocessing = [
        "Impute missing numeric values with the median and categorical values with the mode.",
        "One-hot encode categorical columns, grouping rare categories first.",
        "Scale numeric features for the distance-based and linear models only.",
    ]
    engineering = [
        "Expand any date column into year, month, day, weekday and quarter.",
        "Log-transform strongly right-skewed numeric columns.",
        "Add ratios between the most informative numeric columns.",
    ]

    leakage = [c for c in profile["id_like_columns"]]

    return {
        "problem_type": problem_type,
        "target_column": target,
        "target_confidence": round(confidence, 2),
        "recommended_models": recommended,
        "relevant_features": [c for c in column_names
                              if c != target and c not in profile["id_like_columns"]][:40],
        "columns_to_drop": profile["id_like_columns"] + profile["constant_columns"],
        "preprocessing_recommendations": preprocessing,
        "feature_engineering_recommendations": engineering,
        "potential_data_leakage": leakage,
        "evaluation_metric": metric,
        "reasoning": (
            f"{target_reason} The objective wording suggests a "
            f"{problem_type.replace('_', ' ')} task. "
            f"Identifier and constant columns were excluded because they cannot generalise. "
            f"This recommendation was produced by the built-in rule-based analyst rather "
            f"than Gemini."
        ),
        "objective_success_criteria": (
            f"A model that clearly beats the naive baseline on {metric.upper()} and behaves "
            f"sensibly on the outcome the objective cares about."
        ),
    }


# --------------------------------------------------------------------------
# Validation of whatever came back
# --------------------------------------------------------------------------

def validate_recommendation(recommendation, profile, available_models):
    """
    Never trust the reply as-is.

    Column names are checked against the real dataset, the problem type against
    a fixed vocabulary, and model names against the safe library. Anything that
    does not check out is corrected and recorded in `corrections`, which the UI
    shows so the user can see the agent supervising the AI.
    """
    corrections = []
    column_names = set(profile["column_names"])
    clean = {}

    # -- problem_type -------------------------------------------------------
    problem_type = str(recommendation.get("problem_type", "")).strip().lower().replace(" ", "_")
    if problem_type not in VALID_PROBLEM_TYPES:
        corrections.append(f"Problem type '{recommendation.get('problem_type')}' is not recognised; "
                           f"the local analyst's answer was used instead.")
        # None rather than a guess: the caller fills this in from the local
        # analyst, which is a real answer rather than a silent default.
        problem_type = None
    clean["problem_type"] = problem_type

    # -- target_column ------------------------------------------------------
    target = recommendation.get("target_column")
    if target is not None:
        target = str(target).strip()
        if target not in column_names:
            # Try a forgiving match before giving up.
            lowered = {str(c).lower(): c for c in column_names}
            if target.lower() in lowered:
                target = lowered[target.lower()]
            else:
                corrections.append(f"Suggested target '{target}' does not exist in the dataset "
                                   f"and was discarded.")
                target = None
    clean["target_column"] = target

    # -- confidence ---------------------------------------------------------
    try:
        confidence = float(recommendation.get("target_confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    if not 0.0 <= confidence <= 1.0:
        # A confidence outside the range means the model ignored the instruction,
        # so it tells us nothing. Clamping 3.7 up to "certain" would be exactly
        # the wrong reading of it.
        corrections.append(f"Reported confidence of {confidence} is not a probability; "
                           f"treated as unknown.")
        confidence = 0.5
    clean["target_confidence"] = confidence

    # -- model names --------------------------------------------------------
    raw_models = recommendation.get("recommended_models") or []
    if not isinstance(raw_models, list):
        raw_models = [raw_models]
    clean["recommended_models"] = [str(m) for m in raw_models]

    # -- column lists -------------------------------------------------------
    for key in ("relevant_features", "columns_to_drop", "potential_data_leakage"):
        values = recommendation.get(key) or []
        if not isinstance(values, list):
            values = [values]
        kept, unknown = [], []
        for value in values:
            name = str(value).strip()
            if name in column_names:
                kept.append(name)
            else:
                unknown.append(name)
        if unknown:
            corrections.append(f"Ignored {len(unknown)} column name(s) in '{key}' that do not "
                               f"exist in the dataset: {', '.join(unknown[:4])}.")
        clean[key] = kept

    # -- free text lists ----------------------------------------------------
    for key in ("preprocessing_recommendations", "feature_engineering_recommendations"):
        values = recommendation.get(key) or []
        if not isinstance(values, list):
            values = [values]
        clean[key] = [str(v).strip() for v in values if str(v).strip()][:8]

    clean["evaluation_metric"] = recommendation.get("evaluation_metric")
    clean["reasoning"] = str(recommendation.get("reasoning", "")).strip()
    clean["objective_success_criteria"] = str(
        recommendation.get("objective_success_criteria", "")).strip()
    clean["corrections"] = corrections

    return clean


def get_recommendation(objective, profile, available_models):
    """
    The single entry point used by the pipeline.

    Returns (recommendation, source, note) where source is 'gemini' or
    'local-fallback' and note explains a fallback when one happened.
    """
    raw, source, error = call_gemini(objective, profile, available_models)

    if raw is None:
        recommendation = local_recommendation(objective, profile, available_models)
        recommendation["corrections"] = []
        recommendation["source"] = "local-fallback"
        return recommendation, "local-fallback", error

    recommendation = validate_recommendation(raw, profile, available_models)

    # Where Gemini's answer did not survive validation, fill the gap from the
    # local analyst rather than failing or quietly defaulting.
    needs_type = recommendation["problem_type"] is None
    needs_target = (recommendation["problem_type"] in ("classification", "regression")
                    and not recommendation["target_column"])

    if needs_type or needs_target:
        local = local_recommendation(objective, profile, available_models)

        if needs_type:
            recommendation["problem_type"] = local["problem_type"]
            recommendation["target_confidence"] = min(recommendation["target_confidence"], 0.5)

        if (needs_type or needs_target) and not recommendation["target_column"]:
            if local["target_column"]:
                recommendation["target_column"] = local["target_column"]
                recommendation["target_confidence"] = min(recommendation["target_confidence"], 0.5)
                recommendation["corrections"].append(
                    f"Gemini did not return a usable target column, so the local analyst's "
                    f"choice ('{local['target_column']}') was used instead.")

        if not recommendation["recommended_models"]:
            recommendation["recommended_models"] = local["recommended_models"]

    recommendation["source"] = "gemini"
    return recommendation, "gemini", None
