"""
The safe model library.

This is the security boundary of the whole application. Gemini is allowed to
*name* models it thinks are suitable; it is never allowed to produce code. Every
name it returns is resolved against this dictionary, and anything that does not
resolve is ignored. The only estimators that can ever run are the ones written
out below by hand.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42


def _model(build, task, needs_scaling=False, max_rows=None, is_baseline=False, note=""):
    return {
        "build": build,
        "task": task,
        "needs_scaling": needs_scaling,
        "max_rows": max_rows,        # skip this model on datasets larger than this
        "is_baseline": is_baseline,
        "note": note,
    }


# --------------------------------------------------------------------------
# The library. Sixteen real models plus one baseline per task.
# --------------------------------------------------------------------------

MODEL_LIBRARY = {
    # ---------------------------- classification --------------------------
    "Logistic Regression": _model(
        lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "classification", needs_scaling=True,
        note="Linear, fast and highly interpretable - the sensible starting point."),

    "Decision Tree Classifier": _model(
        lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "classification",
        note="Captures non-linear rules and is easy to explain to a non-technical audience."),

    "Random Forest Classifier": _model(
        lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                       n_jobs=-1, random_state=RANDOM_STATE),
        "classification",
        note="Averaging many trees controls the overfitting a single tree suffers from."),

    "Extra Trees Classifier": _model(
        lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "classification",
        note="Like a random forest but with extra randomness in the splits, often faster."),

    "K-Nearest Neighbors Classifier": _model(
        lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "classification", needs_scaling=True, max_rows=25000,
        note="Predicts from the most similar historical rows. Needs scaled features."),

    "Gaussian Naive Bayes": _model(
        lambda: GaussianNB(),
        "classification",
        note="Very fast probabilistic baseline; assumes features are independent."),

    "Support Vector Classifier": _model(
        lambda: SVC(probability=True, random_state=RANDOM_STATE),
        "classification", needs_scaling=True, max_rows=10000,
        note="Strong on small, clean datasets but scales poorly with row count."),

    "Gradient Boosting Classifier": _model(
        lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        "classification",
        note="Builds trees sequentially, each correcting the previous one's errors."),

    "HistGradientBoosting Classifier": _model(
        lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "classification",
        note="A histogram-based booster that stays fast on larger datasets."),

    # ------------------------------ regression ----------------------------
    "Linear Regression": _model(
        lambda: LinearRegression(),
        "regression", needs_scaling=True,
        note="The reference model: a straight-line relationship with readable coefficients."),

    "Ridge Regression": _model(
        lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "regression", needs_scaling=True,
        note="Linear regression with a penalty that keeps correlated features stable."),

    "Lasso Regression": _model(
        lambda: Lasso(alpha=0.01, max_iter=5000, random_state=RANDOM_STATE),
        "regression", needs_scaling=True,
        note="Shrinks weak coefficients to exactly zero, performing its own feature selection."),

    "Decision Tree Regressor": _model(
        lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "regression",
        note="Splits the data into regions and predicts the average of each region."),

    "Random Forest Regressor": _model(
        lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=-1, random_state=RANDOM_STATE),
        "regression",
        note="Usually the strongest all-round choice on tabular regression problems."),

    "Extra Trees Regressor": _model(
        lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                    n_jobs=-1, random_state=RANDOM_STATE),
        "regression",
        note="A faster, higher-variance cousin of the random forest."),

    "Gradient Boosting Regressor": _model(
        lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
        "regression",
        note="Sequential boosting, often the most accurate model on structured data."),

    "HistGradientBoosting Regressor": _model(
        lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "regression",
        note="Histogram-based boosting that handles larger datasets comfortably."),

    "K-Nearest Neighbors Regressor": _model(
        lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "regression", needs_scaling=True, max_rows=25000,
        note="Averages the target of the most similar historical rows."),

    # ------------------------------- baselines ----------------------------
    "Baseline (Most Frequent Class)": _model(
        lambda: DummyClassifier(strategy="most_frequent"),
        "classification", is_baseline=True,
        note="Always predicts the majority class. Any real model must beat this."),

    "Baseline (Mean Prediction)": _model(
        lambda: DummyRegressor(strategy="mean"),
        "regression", is_baseline=True,
        note="Always predicts the average value. Any real model must beat this."),
}


# Names Gemini (or a user) might plausibly use for each model in the library.
MODEL_ALIASES = {
    "logistic regression": "Logistic Regression",
    "logisticregression": "Logistic Regression",
    "logit": "Logistic Regression",
    "decision tree": "Decision Tree Classifier",
    "decision tree classifier": "Decision Tree Classifier",
    "decisiontreeclassifier": "Decision Tree Classifier",
    "cart": "Decision Tree Classifier",
    "random forest": "Random Forest Classifier",
    "random forest classifier": "Random Forest Classifier",
    "randomforestclassifier": "Random Forest Classifier",
    "rf": "Random Forest Classifier",
    "extra trees": "Extra Trees Classifier",
    "extra trees classifier": "Extra Trees Classifier",
    "extratreesclassifier": "Extra Trees Classifier",
    "knn": "K-Nearest Neighbors Classifier",
    "k-nearest neighbors": "K-Nearest Neighbors Classifier",
    "k nearest neighbors": "K-Nearest Neighbors Classifier",
    "kneighborsclassifier": "K-Nearest Neighbors Classifier",
    "naive bayes": "Gaussian Naive Bayes",
    "gaussian naive bayes": "Gaussian Naive Bayes",
    "gaussiannb": "Gaussian Naive Bayes",
    "svm": "Support Vector Classifier",
    "svc": "Support Vector Classifier",
    "support vector machine": "Support Vector Classifier",
    "support vector classifier": "Support Vector Classifier",
    "gradient boosting": "Gradient Boosting Classifier",
    "gradient boosting classifier": "Gradient Boosting Classifier",
    "gradientboostingclassifier": "Gradient Boosting Classifier",
    "histgradientboosting": "HistGradientBoosting Classifier",
    "hist gradient boosting": "HistGradientBoosting Classifier",
    # Models we deliberately do not ship, mapped to their closest safe cousin.
    "xgboost": "HistGradientBoosting Classifier",
    "xgb": "HistGradientBoosting Classifier",
    "lightgbm": "HistGradientBoosting Classifier",
    "lgbm": "HistGradientBoosting Classifier",
    "catboost": "HistGradientBoosting Classifier",

    "linear regression": "Linear Regression",
    "linearregression": "Linear Regression",
    "ols": "Linear Regression",
    "ridge": "Ridge Regression",
    "ridge regression": "Ridge Regression",
    "lasso": "Lasso Regression",
    "lasso regression": "Lasso Regression",
    "elastic net": "Ridge Regression",
    "elasticnet": "Ridge Regression",
    "decision tree regressor": "Decision Tree Regressor",
    "decisiontreeregressor": "Decision Tree Regressor",
    "random forest regressor": "Random Forest Regressor",
    "randomforestregressor": "Random Forest Regressor",
    "extra trees regressor": "Extra Trees Regressor",
    "extratreesregressor": "Extra Trees Regressor",
    "gradient boosting regressor": "Gradient Boosting Regressor",
    "gradientboostingregressor": "Gradient Boosting Regressor",
    "histgradientboosting regressor": "HistGradientBoosting Regressor",
    "knn regressor": "K-Nearest Neighbors Regressor",
    "kneighborsregressor": "K-Nearest Neighbors Regressor",
}

# When Gemini is unavailable or unhelpful, train this reliable spread instead.
DEFAULT_CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "K-Nearest Neighbors Classifier",
    "Gaussian Naive Bayes",
]

DEFAULT_REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
]

BASELINE_FOR_TASK = {
    "classification": "Baseline (Most Frequent Class)",
    "regression": "Baseline (Mean Prediction)",
}


def normalise_model_name(name):
    """
    Map a free-text model name onto a key in MODEL_LIBRARY, or return None.

    Returning None is the safe outcome: the caller simply ignores the
    suggestion rather than trying to run something that does not exist.
    """
    if not name or not isinstance(name, str):
        return None

    candidate = name.strip()
    if candidate in MODEL_LIBRARY:
        return candidate

    lowered = candidate.lower().strip()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]

    # Case-insensitive exact match against the library keys.
    for key in MODEL_LIBRARY:
        if key.lower() == lowered:
            return key

    return None


def resolve_models(suggested_names, task, dataset_rows):
    """
    Turn a list of suggested model names into a validated, task-compatible,
    size-appropriate training plan.

    Returns (selected, rejected) where `rejected` explains every name we
    refused, so the UI can show that the agent did not silently ignore the AI.
    """
    selected = []
    rejected = []
    seen = set()

    for raw_name in (suggested_names or []):
        resolved = normalise_model_name(raw_name)
        if resolved is None:
            rejected.append({
                "name": str(raw_name),
                "reason": "Not present in the safe model library, so it was ignored.",
            })
            continue

        spec = MODEL_LIBRARY[resolved]
        if spec["task"] != task:
            rejected.append({
                "name": str(raw_name),
                "reason": f"It is a {spec['task']} model, but this is a {task} problem.",
            })
            continue
        if spec["max_rows"] and dataset_rows > spec["max_rows"]:
            rejected.append({
                "name": resolved,
                "reason": f"Skipped because the dataset has {dataset_rows:,} rows and this "
                          f"model becomes impractically slow above {spec['max_rows']:,}.",
            })
            continue
        if resolved in seen:
            continue

        seen.add(resolved)
        selected.append(resolved)

    # Guarantee a sensible spread even if the AI suggested nothing usable.
    defaults = (DEFAULT_CLASSIFICATION_MODELS if task == "classification"
                else DEFAULT_REGRESSION_MODELS)
    if len(selected) < 3:
        for name in defaults:
            if name in seen:
                continue
            spec = MODEL_LIBRARY[name]
            if spec["max_rows"] and dataset_rows > spec["max_rows"]:
                continue
            seen.add(name)
            selected.append(name)
            if len(selected) >= 5:
                break

    # Always train the baseline, so "is the model actually any good?" has an answer.
    baseline = BASELINE_FOR_TASK.get(task)
    if baseline and baseline not in seen:
        selected.append(baseline)

    return selected, rejected


def list_library(task=None):
    """Used by the UI to show which models the agent is allowed to run."""
    return [
        {
            "name": name,
            "task": spec["task"],
            "needs_scaling": spec["needs_scaling"],
            "is_baseline": spec["is_baseline"],
            "note": spec["note"],
        }
        for name, spec in MODEL_LIBRARY.items()
        if task is None or spec["task"] == task
    ]
