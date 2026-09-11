"""
Preprocessing, training, evaluation, best-model selection and - the part that
actually matters - deciding whether the user's objective was met.

A deliberate distinction runs through this module: a model can score well and
still fail the objective (for example a churn model with 94% accuracy that
never once predicts "churn"). Sections `pick_best_model` and
`validate_objective` keep those two questions separate.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agent import models as model_library

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}
REGRESSION_METRICS = {"r2", "rmse", "mae", "mse"}

# Metrics where a smaller number is better.
LOWER_IS_BETTER = {"rmse", "mae", "mse"}


# --------------------------------------------------------------------------
# Problem type
# --------------------------------------------------------------------------

def detect_problem_type(target_series):
    """
    Decide classification vs regression from the target column itself.

    This is the local check that Gemini's answer has to survive. The data wins
    any disagreement: if the target is text, no amount of AI confidence makes
    it a regression problem.
    """
    series = target_series.dropna()
    unique = series.nunique()

    if unique <= 1:
        return "invalid", f"The target has only {unique} distinct value(s), so there is nothing to predict."

    if not pd.api.types.is_numeric_dtype(series):
        return "classification", f"The target is non-numeric with {unique} distinct classes."

    if pd.api.types.is_bool_dtype(series) or unique == 2:
        return "classification", "The target has exactly two distinct values, so this is binary classification."

    # Whole numbers with few distinct values are labels wearing numeric clothing.
    looks_integer = np.allclose(series.dropna() % 1, 0)
    if looks_integer and unique <= 20 and unique / max(len(series), 1) < 0.05:
        return "classification", (f"The target is numeric but takes only {unique} whole-number "
                                  f"values, which is a class label rather than a quantity.")

    return "regression", f"The target is continuous with {unique:,} distinct numeric values."


def choose_metric(problem_type, target_series, suggested=None):
    """Pick the headline metric, honouring a valid AI suggestion where possible."""
    allowed = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS

    if suggested:
        cleaned = str(suggested).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "f1_score": "f1", "f1score": "f1", "f_1": "f1", "weighted_f1": "f1",
            "auc": "roc_auc", "aucroc": "roc_auc", "roc": "roc_auc", "auc_roc": "roc_auc",
            "r2_score": "r2", "r_squared": "r2", "rsquared": "r2",
            "root_mean_squared_error": "rmse", "mean_absolute_error": "mae",
            "mean_squared_error": "mse",
        }
        cleaned = aliases.get(cleaned, cleaned)
        if cleaned in allowed:
            return cleaned, f"Using {cleaned.upper()} as recommended by the AI analyst for this objective."

    if problem_type == "classification":
        counts = target_series.dropna().value_counts(normalize=True)
        minority_share = counts.min() if len(counts) else 0.5
        if len(counts) == 2 and minority_share < 0.35:
            return "roc_auc", (f"The classes are imbalanced (the smaller class is only "
                               f"{minority_share:.1%} of rows), so accuracy would be misleading. "
                               f"ROC-AUC measures how well the model separates the classes at "
                               f"every threshold.")
        if len(counts) == 2:
            return "f1", ("The classes are reasonably balanced, and F1 balances precision against "
                          "recall rather than rewarding a model for playing it safe.")
        return "f1", (f"Multi-class problem with {len(counts)} classes; weighted F1 accounts for "
                      f"class sizes instead of letting the largest class dominate.")

    return "r2", ("R² reports the share of variation in the target the model explains, which is "
                  "comparable across datasets in a way that RMSE is not.")


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def build_preprocessor(features, needs_scaling):
    """
    Build the ColumnTransformer that turns a mixed dataframe into a numeric
    matrix: median-impute + optionally scale the numbers, most-frequent-impute
    + one-hot encode the categories.

    Everything is dense on purpose - GaussianNB and the histogram boosters
    cannot consume sparse matrices, and at this data scale density costs
    nothing.
    """
    numeric_columns = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical_columns = [c for c in features.columns if c not in numeric_columns]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if needs_scaling:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
    ]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(features, target, problem_type):
    """Split into train and test, stratifying classification targets when possible."""
    stratify = None
    note = f"Random {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split with random_state={RANDOM_STATE}."

    if problem_type == "classification":
        counts = target.value_counts()
        if counts.min() >= 2 and len(counts) < len(target) * 0.5:
            stratify = target
            note = (f"Stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split with "
                    f"random_state={RANDOM_STATE}, so both splits keep the same class balance.")

    features_train, features_test, target_train, target_test = train_test_split(
        features, target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return features_train, features_test, target_train, target_test, note


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _classification_metrics(y_true, y_pred, y_proba, classes):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr",
                                  average="weighted", labels=classes))
        except (ValueError, IndexError):
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["classes"] = [str(c) for c in classes]

    # Per-class recall makes "the model never predicts the rare class" visible.
    per_class = {}
    for index, label in enumerate(classes):
        support = int(matrix[index].sum())
        correct = int(matrix[index][index])
        per_class[str(label)] = {
            "support": support,
            "recall": round(correct / support, 4) if support else 0.0,
        }
    metrics["per_class"] = per_class
    return metrics


def _regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_one_model(name, features_train, target_train, features_test, target_test,
                    problem_type, metric):
    """
    Train a single model and return a result dictionary.

    A failure here is data, not a crash: the reason is recorded and the agent
    carries on with the remaining models.
    """
    spec = model_library.MODEL_LIBRARY[name]
    started = time.time()

    result = {
        "model_name": name,
        "status": "trained",
        "is_baseline": spec["is_baseline"],
        "note": spec["note"],
        "primary_metric": metric,
        "primary_score": None,
        "metrics": {},
        "params": {},
        "error_message": None,
        "train_time_sec": None,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(features_train, spec["needs_scaling"])),
                ("model", spec["build"]()),
            ])
            pipeline.fit(features_train, target_train)
            predictions = pipeline.predict(features_test)

            if problem_type == "classification":
                probabilities = None
                if hasattr(pipeline.named_steps["model"], "predict_proba"):
                    try:
                        probabilities = pipeline.predict_proba(features_test)
                    except Exception:
                        probabilities = None
                classes = list(pipeline.named_steps["model"].classes_)
                result["metrics"] = _classification_metrics(
                    target_test, predictions, probabilities, classes)
            else:
                result["metrics"] = _regression_metrics(target_test, predictions)

        score = result["metrics"].get(metric)
        result["primary_score"] = float(score) if score is not None else None
        result["params"] = {
            k: str(v) for k, v in spec["build"]().get_params().items()
            if k in ("n_estimators", "max_depth", "alpha", "n_neighbors", "C",
                     "min_samples_leaf", "max_iter", "strategy", "learning_rate")
        }
        result["pipeline"] = pipeline
        result["predictions"] = predictions

    except Exception as exc:                      # noqa: BLE001 - we want every failure
        result["status"] = "failed"
        result["error_message"] = f"{type(exc).__name__}: {exc}"[:500]

    result["train_time_sec"] = round(time.time() - started, 3)
    return result


def train_models(model_names, features_train, target_train, features_test, target_test,
                 problem_type, metric, progress_callback=None):
    """Train every selected model in turn, never letting one failure stop the run."""
    results = []
    for position, name in enumerate(model_names):
        if progress_callback:
            progress_callback(position + 1, len(model_names), name)
        results.append(train_one_model(
            name, features_train, target_train, features_test, target_test,
            problem_type, metric))
    return results


# --------------------------------------------------------------------------
# Best model
# --------------------------------------------------------------------------

def pick_best_model(results, metric):
    """
    Choose the winner on the objective-appropriate metric - never blindly on
    accuracy - and exclude the baseline, which exists only as a yardstick.
    """
    usable = [r for r in results
              if r["status"] == "trained" and r["primary_score"] is not None and not r["is_baseline"]]
    if not usable:
        # If every real model failed, fall back to the baseline so the run still
        # produces an honest answer rather than nothing at all.
        usable = [r for r in results if r["status"] == "trained" and r["primary_score"] is not None]
    if not usable:
        return None

    reverse = metric not in LOWER_IS_BETTER
    usable.sort(key=lambda r: r["primary_score"], reverse=reverse)
    best = usable[0]
    best["is_best"] = True

    runner_up = usable[1] if len(usable) > 1 else None
    if runner_up:
        gap = abs(best["primary_score"] - runner_up["primary_score"])
        best["selection_reason"] = (
            f"{best['model_name']} achieved the best {metric.upper()} of "
            f"{best['primary_score']:.4f}, ahead of {runner_up['model_name']} "
            f"({runner_up['primary_score']:.4f}) by {gap:.4f}. "
            f"{metric.upper()} was chosen because it matches what the objective is asking for."
        )
    else:
        best["selection_reason"] = (
            f"{best['model_name']} was the only model that trained successfully, with a "
            f"{metric.upper()} of {best['primary_score']:.4f}."
        )
    return best


def get_baseline_score(results, metric):
    for result in results:
        if result["is_baseline"] and result["status"] == "trained":
            return result["primary_score"]
    return None


def cross_validate_best(best_result, features, target, problem_type, metric):
    """
    Re-score the winner with 5-fold cross-validation on smaller datasets.

    A single test split can be lucky. If the cross-validated score is far below
    the test score, the model is less stable than it looks, and the report says so.
    """
    if best_result is None or best_result["status"] != "trained" or len(features) > 20000:
        return None

    scoring = {
        "accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted",
        "recall": "recall_weighted", "roc_auc": "roc_auc_ovr_weighted",
        "r2": "r2", "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
    }.get(metric)
    if scoring is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = model_library.MODEL_LIBRARY[best_result["model_name"]]
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(features, spec["needs_scaling"])),
                ("model", spec["build"]()),
            ])
            scores = cross_val_score(pipeline, features, target, cv=5, scoring=scoring,
                                     error_score="raise")
    except Exception:
        return None

    values = np.abs(scores) if scoring.startswith("neg_") else scores
    return {
        "folds": 5,
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "scores": [round(float(v), 4) for v in values],
    }


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------

def _feature_names(pipeline):
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return []


def extract_feature_importance(best_result, features_test, target_test, problem_type, top_n=15):
    """
    Explain the winning model.

    Tree models expose impurity-based importances and linear models expose
    coefficients. For everything else (KNN, SVC, Naive Bayes) we fall back to
    permutation importance, which asks the only question that always makes
    sense: how much worse does the model get if this column is shuffled?
    """
    if best_result is None or best_result["status"] != "trained":
        return {"method": None, "features": []}

    pipeline = best_result.get("pipeline")
    if pipeline is None:
        return {"method": None, "features": []}

    estimator = pipeline.named_steps["model"]
    names = _feature_names(pipeline)

    values, method, explanation = None, None, ""

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "Impurity-based importance"
        explanation = ("How much each feature reduced prediction error across all the splits "
                       "in the trees. Higher means the model relied on it more.")
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        method = "Absolute model coefficients"
        explanation = ("The size of each feature's coefficient after scaling. Larger means a "
                       "one-unit change in that feature moves the prediction more.")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample = features_test.head(1500)
                sample_target = target_test.head(1500)
                computed = permutation_importance(
                    pipeline, sample, sample_target,
                    n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
            values = np.asarray(computed.importances_mean, dtype=float)
            names = list(features_test.columns)
            method = "Permutation importance"
            explanation = ("How much the model's score drops when each column is randomly "
                           "shuffled. A big drop means the model genuinely depends on it.")
        except Exception:
            return {"method": None, "features": []}

    if values is None or len(names) != len(values):
        return {"method": method, "features": [], "explanation": explanation}

    frame = pd.DataFrame({"feature": names, "importance": values})
    frame["importance"] = frame["importance"].abs()
    total = frame["importance"].sum()
    frame["share"] = (frame["importance"] / total * 100) if total else 0.0
    frame = frame.sort_values("importance", ascending=False).head(top_n)

    return {
        "method": method,
        "explanation": explanation,
        "features": [
            {
                "feature": str(row.feature).replace("numeric__", "").replace("categorical__", ""),
                "importance": round(float(row.importance), 6),
                "share": round(float(row.share), 2),
            }
            for row in frame.itertuples()
        ],
    }


# --------------------------------------------------------------------------
# Objective validation - the question the whole app exists to answer
# --------------------------------------------------------------------------

def validate_objective(best_result, problem_type, metric, baseline_score, target_series,
                       success_criteria=None, cross_validation=None):
    """
    Decide whether the user's objective was actually met.

    Performance and fulfilment are not the same thing. This function checks
    three things: is the score good in absolute terms, does it beat the naive
    baseline, and does the model behave usefully on the class that matters.
    """
    if best_result is None or best_result["status"] != "trained":
        return {
            "status": "NOT FULFILLED",
            "reason": "No model trained successfully, so the objective could not be evaluated.",
            "checks": [],
        }

    score = best_result["primary_score"]
    metrics = best_result["metrics"]
    checks = []

    # ---- Check 1: absolute quality ---------------------------------------
    if problem_type == "classification":
        strong, acceptable = (0.80, 0.65) if metric != "roc_auc" else (0.85, 0.70)
        if score >= strong:
            checks.append({"name": "Predictive quality", "passed": True,
                           "detail": f"{metric.upper()} of {score:.4f} is strong for this kind of problem."})
        elif score >= acceptable:
            checks.append({"name": "Predictive quality", "passed": "partial",
                           "detail": f"{metric.upper()} of {score:.4f} is usable but leaves real room to improve."})
        else:
            checks.append({"name": "Predictive quality", "passed": False,
                           "detail": f"{metric.upper()} of {score:.4f} is too low to rely on for decisions."})
    else:
        r2 = metrics.get("r2", 0.0)
        if r2 >= 0.70:
            checks.append({"name": "Predictive quality", "passed": True,
                           "detail": f"The model explains {r2:.1%} of the variation in the target."})
        elif r2 >= 0.40:
            checks.append({"name": "Predictive quality", "passed": "partial",
                           "detail": f"The model explains {r2:.1%} of the variation - a real signal, "
                                     f"but a lot is still unexplained."})
        else:
            checks.append({"name": "Predictive quality", "passed": False,
                           "detail": f"The model explains only {r2:.1%} of the variation, which is "
                                     f"too little to be useful."})

    # ---- Check 2: does it beat doing nothing clever? ----------------------
    if baseline_score is not None:
        if metric in LOWER_IS_BETTER:
            improvement = baseline_score - score
            beat = improvement > 0
        else:
            improvement = score - baseline_score
            beat = improvement > 0.02
        checks.append({
            "name": "Beats the naive baseline",
            "passed": bool(beat),
            "detail": (f"The trivial baseline scores {baseline_score:.4f} and the chosen model "
                       f"scores {score:.4f} - an improvement of {abs(improvement):.4f}."
                       if beat else
                       f"The trivial baseline scores {baseline_score:.4f} and the chosen model only "
                       f"{score:.4f}. The model is not adding meaningful value over guessing."),
        })

    # ---- Check 3: is it useful on the class that matters? ----------------
    if problem_type == "classification":
        per_class = metrics.get("per_class", {})
        if per_class:
            counts = target_series.value_counts()
            minority = str(counts.idxmin())
            minority_recall = per_class.get(minority, {}).get("recall", 0.0)
            if minority_recall >= 0.60:
                checks.append({"name": "Catches the minority class", "passed": True,
                               "detail": f"It correctly identifies {minority_recall:.1%} of the "
                                         f"'{minority}' cases, which is the class the objective cares about."})
            elif minority_recall >= 0.30:
                checks.append({"name": "Catches the minority class", "passed": "partial",
                               "detail": f"It finds only {minority_recall:.1%} of the '{minority}' cases. "
                                         f"It is right when it fires, but it misses most of them."})
            else:
                checks.append({"name": "Catches the minority class", "passed": False,
                               "detail": f"It identifies just {minority_recall:.1%} of the '{minority}' "
                                         f"cases, so it is close to useless for the decision this "
                                         f"objective implies."})

    # ---- Check 4: is the result stable? ----------------------------------
    if cross_validation:
        gap = abs(cross_validation["mean"] - score)
        stable = gap <= max(0.05, 2 * cross_validation["std"])
        checks.append({
            "name": "Stable across folds",
            "passed": bool(stable),
            "detail": (f"5-fold cross-validation gives {cross_validation['mean']:.4f} "
                       f"(± {cross_validation['std']:.4f}) against a test score of {score:.4f}, "
                       f"so the result is {'consistent' if stable else 'noticeably unstable'}."),
        })

    # ---- Verdict ----------------------------------------------------------
    failed = [c for c in checks if c["passed"] is False]
    partial = [c for c in checks if c["passed"] == "partial"]

    if failed:
        status = "NOT FULFILLED" if len(failed) > 1 else "PARTIALLY FULFILLED"
    elif partial:
        status = "PARTIALLY FULFILLED"
    else:
        status = "FULFILLED"

    lead = {
        "FULFILLED": "The objective was met.",
        "PARTIALLY FULFILLED": "The objective was partially met.",
        "NOT FULFILLED": "The objective was not met.",
    }[status]

    # The headline reason stays short: the quality sentence, plus the single most
    # important problem if there is one. The individual checks are listed
    # alongside it, so repeating all of them here would only add noise.
    sentences = [checks[0]["detail"]]
    weakest = (failed or partial)
    if weakest and weakest[0] is not checks[0]:
        sentences.append(weakest[0]["detail"])

    reason = f"{lead} {' '.join(sentences)}"
    if success_criteria:
        reason += f" Success criterion: {success_criteria}"

    return {"status": status, "reason": reason, "checks": checks}
