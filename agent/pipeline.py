"""
The orchestrator.

`run_analysis` walks the full data-science workflow in order and reports real
progress as it goes. It is written as one readable top-to-bottom function on
purpose: an interviewer should be able to read it once and know exactly what
the agent does and in what order.

It has no idea that MySQL or FastAPI exist. It takes a dataframe and an
objective, calls a progress callback, and returns a dictionary. That is what
makes it testable on its own.
"""

import pandas as pd

import config
from agent import charts as chart_builder
from agent import cleaning, features, gemini
from agent import models as model_library
from agent import profiling, training

STEPS = [
    ("profile", "Dataset analysed"),
    ("clean", "Data cleaned"),
    ("ai", "Objective analysed by the AI layer"),
    ("problem", "Problem type detected"),
    ("target", "Target variable confirmed"),
    ("engineer", "Features engineered"),
    ("select", "Features selected"),
    ("choose_models", "Models selected"),
    ("train", "Models trained"),
    ("evaluate", "Models evaluated"),
    ("best", "Best model selected"),
    ("validate", "Objective validated"),
    ("charts", "Dashboard generated"),
    ("report", "Report ready"),
]

# Below this confidence the agent stops and asks the user which column to predict
# instead of guessing and quietly modelling the wrong thing.
TARGET_CONFIDENCE_THRESHOLD = 0.60

UNSUPPORTED_EXPLANATIONS = {
    "clustering": (
        "This objective describes clustering - finding natural groups without a known "
        "answer to learn from. This agent automates supervised learning (classification "
        "and regression), where every row has a known outcome. Rather than invent a "
        "result, the analysis stops here with a full data understanding report. "
        "To use the modelling engine, restate the objective around a column you want to "
        "predict."
    ),
    "anomaly_detection": (
        "This objective describes anomaly detection, which needs unsupervised or "
        "semi-supervised methods that this agent does not implement. The data "
        "understanding, quality and cleaning results below are still valid. If your "
        "dataset has a column that labels the anomalies, restate the objective as "
        "predicting that column and the full modelling workflow will run."
    ),
    "time_series_forecasting": (
        "This objective describes time-series forecasting, which needs time-aware "
        "validation (training on the past and testing on the future) rather than the "
        "random split used here. Running the standard workflow would produce optimistic "
        "scores that would not survive contact with real future data, so it has been "
        "stopped. The data understanding results below are still valid."
    ),
}


class Progress:
    """
    A tiny progress reporter.

    The web app passes in a version that writes to MySQL; tests pass in one that
    prints. Keeping it this small is what stops the pipeline from depending on
    the database.
    """

    def __init__(self, on_start=None, on_finish=None, on_update=None):
        self.on_start = on_start
        self.on_finish = on_finish
        self.on_update = on_update

    def start(self, key):
        if self.on_start:
            self.on_start(key)

    def update(self, key, detail):
        """Refresh the detail line of a step that is still running."""
        if self.on_update:
            self.on_update(key, detail)

    def done(self, key, detail=None):
        if self.on_finish:
            self.on_finish(key, detail, "done")

    def skip(self, key, detail=None):
        if self.on_finish:
            self.on_finish(key, detail, "skipped")

    def fail(self, key, detail=None):
        if self.on_finish:
            self.on_finish(key, detail, "failed")


def _strip_unserialisable(results):
    """Remove fitted estimators and raw predictions before the results are stored."""
    cleaned = []
    for result in results:
        copy = {k: v for k, v in result.items() if k not in ("pipeline", "predictions")}
        cleaned.append(copy)
    return cleaned


def _build_insights(profile_after, cleaning_report, selection, best, validation,
                    importance, problem_type, metric):
    """Turn the numbers into sentences a non-technical reader can act on."""
    insights = []

    if cleaning_report["rows_removed"] or cleaning_report["columns_removed"]:
        insights.append(
            f"Cleaning removed {cleaning_report['rows_removed']:,} row(s) and "
            f"{cleaning_report['columns_removed']} column(s) that could not contribute to a "
            f"reliable model."
        )

    if selection.get("leakage"):
        names = ", ".join(item["feature"] for item in selection["leakage"])
        insights.append(
            f"Potential target leakage was detected and removed in: {names}. Left in place, "
            f"these would have produced an impressive score that collapses in production."
        )

    if importance and importance.get("features"):
        top = importance["features"][:3]
        names = ", ".join(f"{f['feature']} ({f['share']:.0f}%)" for f in top)
        insights.append(f"The winning model's decisions are driven mainly by {names}.")

    if best and problem_type == "classification":
        per_class = best.get("metrics", {}).get("per_class", {})
        weakest = min(per_class.items(), key=lambda item: item[1]["recall"]) if per_class else None
        if weakest and weakest[1]["recall"] < 0.6:
            insights.append(
                f"The model is weakest on the '{weakest[0]}' class, catching only "
                f"{weakest[1]['recall']:.0%} of those cases. If that class is the one the "
                f"business cares about, this is the number to improve."
            )

    if best and problem_type == "regression":
        mae = best.get("metrics", {}).get("mae")
        if mae is not None:
            insights.append(
                f"On average the prediction is off by {mae:,.2f} in the target's own units, "
                f"which is the number to quote when someone asks how accurate it is."
            )

    if validation:
        insights.append(validation["reason"])

    return insights


def _build_recommendations(profile_after, validation, problem_type, best, metric, selection):
    """Concrete next steps, phrased as things a person would actually do."""
    recommendations = []

    if profile_after["n_rows"] < 1000:
        recommendations.append(
            f"Collect more data. {profile_after['n_rows']:,} rows is a small sample, and the "
            f"score above could move considerably on a larger one."
        )

    if validation and validation["status"] != "FULFILLED":
        if problem_type == "classification":
            recommendations.append(
                "Try adjusting the decision threshold rather than the model. Moving it away "
                "from 0.5 trades precision for recall and often fixes a model that is accurate "
                "but never flags the class you care about."
            )
            recommendations.append(
                "If the classes are imbalanced, resampling the training data or class weighting "
                "is usually a bigger win than switching algorithms."
            )
        else:
            recommendations.append(
                "Look for missing explanatory variables. When R² is low, the cause is usually "
                "that the drivers of the target were never recorded, not that the model is wrong."
            )

    if best and not best.get("is_baseline"):
        recommendations.append(
            f"Tune {best['model_name']} with a grid or randomised search. This run used sensible "
            f"defaults with no hyperparameter search, so there is headroom left."
        )

    if selection.get("removed"):
        recommendations.append(
            f"Review the {len(selection['removed'])} removed feature(s) with a domain expert. "
            f"Automated selection is statistical; it does not know which columns matter to the business."
        )

    recommendations.append(
        "Before deploying, re-test on data from a later time period than the training data. "
        "A random split cannot tell you whether the pattern holds next quarter."
    )
    return recommendations


# --------------------------------------------------------------------------
# The main workflow
# --------------------------------------------------------------------------

def run_analysis(dataframe, objective, progress=None, forced_target=None,
                 analysis_id=0, chart_dir=None):
    """
    Run the full agent workflow.

    Returns a dictionary whose "status" is one of:
      completed      - a model was trained and evaluated
      needs_target   - the agent is not confident enough to pick a target alone
      exploratory    - the objective does not describe a prediction task
      unsupported    - a valid but out-of-scope task type (clustering, forecasting)
      failed         - something went wrong; "error" explains what
    """
    progress = progress or Progress()
    chart_dir = chart_dir or config.CHART_DIR
    warnings_list = []

    # ---------------------------------------------------- 1. understand ----
    progress.start("profile")
    profile_before = profiling.profile_dataframe(dataframe, sample_rows=config.GEMINI_SAMPLE_ROWS)
    progress.done("profile",
                  f"{profile_before['n_rows']:,} rows × {profile_before['n_columns']} columns, "
                  f"data quality score {profile_before['quality_score']}/100")

    # ------------------------------------------------------- 2. clean ------
    progress.start("clean")
    cleaned_frame, cleaning_report = cleaning.clean_dataframe(
        dataframe, profile_before, protect=[forced_target] if forced_target else None)
    profile_after = profiling.profile_dataframe(cleaned_frame, sample_rows=config.GEMINI_SAMPLE_ROWS)
    progress.done("clean",
                  f"{len(cleaning_report['steps'])} cleaning action(s); "
                  f"{cleaning_report['rows_removed']:,} rows and "
                  f"{cleaning_report['columns_removed']} columns removed")

    if cleaned_frame.shape[1] == 0 or cleaned_frame.shape[0] < 10:
        progress.fail("clean", "Not enough usable data survived cleaning.")
        return {
            "status": "failed",
            "error": ("After cleaning, too little usable data remained to analyse "
                      f"({cleaned_frame.shape[0]} rows × {cleaned_frame.shape[1]} columns). "
                      "This usually means the file is mostly identifiers, empty columns or "
                      "free text."),
            "profile_before": profile_before,
            "cleaning": cleaning_report,
        }

    # ---------------------------------------------- 3. ask the AI layer ----
    progress.start("ai")
    available_models = [name for name, spec in model_library.MODEL_LIBRARY.items()
                        if not spec["is_baseline"]]
    recommendation, ai_source, ai_note = gemini.get_recommendation(
        objective, profile_after, available_models)
    progress.done("ai",
                  f"Recommendation received from "
                  f"{'Gemini' if ai_source == 'gemini' else 'the built-in rule-based analyst'}")
    if ai_note:
        warnings_list.append(f"Gemini was not used for this run: {ai_note}")

    # ------------------------------------------- 4. decide the problem -----
    progress.start("problem")
    problem_type = recommendation["problem_type"]

    if problem_type in UNSUPPORTED_EXPLANATIONS:
        progress.done("problem", f"Detected as {problem_type.replace('_', ' ')} - out of scope")
        for key, _ in STEPS[STEPS.index(("target", "Target variable confirmed")):]:
            progress.skip(key, "Not applicable to this objective.")
        chart_files = chart_builder.generate_all(
            analysis_id, chart_dir, profile=profile_after, dataframe=cleaned_frame)
        return {
            "status": "unsupported",
            "objective": objective,
            "problem_type": problem_type,
            "explanation": UNSUPPORTED_EXPLANATIONS[problem_type],
            "profile_before": profile_before,
            "profile_after": profile_after,
            "cleaning": cleaning_report,
            "recommendation": recommendation,
            "ai_source": ai_source,
            "charts": chart_files,
            "warnings": warnings_list,
        }

    # ----------------------------------------------- 5. confirm target -----
    progress.start("target")
    target_column = forced_target or recommendation["target_column"]
    confidence = 1.0 if forced_target else recommendation["target_confidence"]

    if target_column is not None and target_column not in cleaned_frame.columns:
        if target_column in dataframe.columns:
            # Cleaning removed it; put it back rather than silently changing the question.
            cleaned_frame[target_column] = dataframe.loc[cleaned_frame.index, target_column] \
                if len(cleaned_frame) == len(dataframe) else dataframe[target_column]
            warnings_list.append(
                f"'{target_column}' was restored after cleaning because it is the target column.")
        else:
            warnings_list.append(
                f"The suggested target '{target_column}' is not present after cleaning.")
            target_column = None

    needs_confirmation = (
        problem_type in ("classification", "regression")
        and (target_column is None or confidence < TARGET_CONFIDENCE_THRESHOLD)
    )

    if needs_confirmation:
        progress.skip("target", "Waiting for the user to confirm which column to predict.")
        modelable = [
            info["name"] for info in profile_after["columns"]
            if info["kind"] in ("numeric", "binary", "boolean", "categorical")
            and info["name"] not in profile_after["id_like_columns"]
        ]
        return {
            "status": "needs_target",
            "objective": objective,
            "reason": (
                f"The agent is only {confidence:.0%} confident about which column your objective "
                f"refers to, and modelling the wrong column would produce a confident, wrong "
                f"answer. Please confirm the column you want to predict."
                if target_column else
                "The agent could not identify which column your objective refers to. Please "
                "select the column you want to predict."
            ),
            "suggested_target": target_column,
            "candidates": profile_after["target_candidates"],
            "selectable_columns": modelable,
            "profile_before": profile_before,
            "profile_after": profile_after,
            "cleaning": cleaning_report,
            "recommendation": recommendation,
            "ai_source": ai_source,
            "warnings": warnings_list,
        }

    if target_column is None:
        problem_type = "exploratory"

    # ------------------------- exploratory path (no prediction target) -----
    if problem_type == "exploratory":
        progress.done("target", "No prediction target - running a descriptive analysis")
        for key, _ in STEPS[STEPS.index(("engineer", "Features engineered")):
                            STEPS.index(("charts", "Dashboard generated"))]:
            progress.skip(key, "Not applicable to a descriptive analysis.")
        progress.start("charts")
        chart_files = chart_builder.generate_all(
            analysis_id, chart_dir, profile=profile_after, dataframe=cleaned_frame)
        progress.done("charts", f"{len(chart_files)} chart(s) generated")
        progress.start("report")
        progress.done("report", "Descriptive report ready")
        return {
            "status": "exploratory",
            "objective": objective,
            "problem_type": "exploratory",
            "explanation": (
                "This objective asks to understand the data rather than to predict a specific "
                "column, so the agent produced a full data understanding and quality report "
                "instead of training models. Name a column to predict and the modelling "
                "workflow will run."
            ),
            "profile_before": profile_before,
            "profile_after": profile_after,
            "cleaning": cleaning_report,
            "recommendation": recommendation,
            "ai_source": ai_source,
            "charts": chart_files,
            "warnings": warnings_list,
        }

    # -------------------------------- validate the type against the data ---
    detected_type, type_reason = training.detect_problem_type(cleaned_frame[target_column])

    if detected_type == "invalid":
        progress.fail("problem", type_reason)
        return {"status": "failed", "error": type_reason,
                "profile_before": profile_before, "cleaning": cleaning_report}

    if detected_type != problem_type:
        warnings_list.append(
            f"The AI layer suggested {problem_type}, but '{target_column}' is "
            f"{type_reason.lower()} The agent trusted the data and treated this as "
            f"{detected_type}."
        )
        problem_type = detected_type

    progress.done("problem", f"{problem_type.title()} - {type_reason}")
    progress.done("target", f"Predicting '{target_column}' ({confidence:.0%} confidence)")

    # ------------------------------------------------ prepare the data -----
    modelling_frame, dropped_for_missing_target = cleaning.drop_rows_missing_target(
        cleaned_frame, target_column)
    if dropped_for_missing_target:
        warnings_list.append(
            f"{dropped_for_missing_target:,} row(s) had no value for '{target_column}' and were "
            f"removed. A missing label cannot be imputed without inventing the answer.")

    sampled = False
    if len(modelling_frame) > config.MAX_TRAINING_ROWS:
        modelling_frame = modelling_frame.sample(
            config.MAX_TRAINING_ROWS, random_state=config.RANDOM_STATE).reset_index(drop=True)
        sampled = True
        warnings_list.append(
            f"The dataset was randomly sampled down to {config.MAX_TRAINING_ROWS:,} rows for "
            f"training so the analysis completes in reasonable time. Profiling used every row.")

    target = modelling_frame[target_column]
    feature_frame = modelling_frame.drop(columns=[target_column])

    # Drop anything the AI flagged for removal, as long as it is not the target.
    ai_drops = [c for c in recommendation.get("columns_to_drop", [])
                if c in feature_frame.columns]
    if ai_drops:
        feature_frame = feature_frame.drop(columns=ai_drops)

    if problem_type == "classification":
        counts = target.value_counts()
        rare_classes = counts[counts < 5].index.tolist()
        if rare_classes and len(counts) > len(rare_classes):
            keep = ~target.isin(rare_classes)
            removed_rows = int((~keep).sum())
            target = target[keep]
            feature_frame = feature_frame[keep]
            warnings_list.append(
                f"{len(rare_classes)} class(es) had fewer than 5 examples and were removed "
                f"({removed_rows} row(s)). A model cannot learn or be fairly tested on a class "
                f"that rare.")

    if len(feature_frame.columns) == 0:
        progress.fail("engineer", "No usable feature columns remain.")
        return {"status": "failed",
                "error": "No usable feature columns remain after cleaning and exclusions.",
                "profile_before": profile_before, "cleaning": cleaning_report}

    if len(feature_frame) < 30:
        warnings_list.append(
            f"Only {len(feature_frame)} rows are available for modelling. Any score reported "
            f"below should be treated as indicative, not reliable.")

    # ---------------------------------- split before touching anything -----
    features_train, features_test, target_train, target_test, split_note = training.split_data(
        feature_frame, target, problem_type)

    # ------------------------------------------ 6. feature engineering -----
    progress.start("engineer")

    # Leakage is checked on the raw columns first. If a leaky column survived
    # into feature engineering, every ratio and difference derived from it
    # would leak too, and those derived copies are much harder to spot later.
    raw_leakage = features.detect_leakage(features_train, target_train, problem_type)
    if raw_leakage:
        leaky_columns = [item["feature"] for item in raw_leakage]
        features_train = features_train.drop(columns=leaky_columns)
        features_test = features_test.drop(columns=leaky_columns)
        feature_frame = feature_frame.drop(columns=leaky_columns)
        warnings_list.append(
            f"Removed {len(leaky_columns)} column(s) as suspected target leakage before "
            f"modelling: {', '.join(leaky_columns)}.")

    recipe = features.plan_feature_engineering(features_train, target_train, problem_type)
    features_train = features.apply_feature_engineering(features_train, recipe)
    features_test = features.apply_feature_engineering(features_test, recipe)
    progress.done("engineer", f"{len(recipe['created'])} derived feature(s) created")

    # -------------------------------------------- 7. feature selection -----
    progress.start("select")
    kept, removed, scores, leakage = features.select_features(
        features_train, target_train, problem_type)
    features_train = features_train[kept]
    features_test = features_test[kept]
    selection = {
        "kept": kept,
        "removed": removed,
        "leakage": raw_leakage + leakage,
        "scores": {str(k): round(float(v), 4) for k, v in scores.items()} if len(scores) else {},
    }
    total_leakage = len(selection["leakage"])
    progress.done("select",
                  f"{len(kept)} feature(s) kept, {len(removed)} removed"
                  + (f", {total_leakage} leakage risk(s) found" if total_leakage else ""))

    # ----------------------------------------------- 8. choose models ------
    progress.start("choose_models")
    metric, metric_reason = training.choose_metric(
        problem_type, target, recommendation.get("evaluation_metric"))
    selected_models, rejected_models = model_library.resolve_models(
        recommendation.get("recommended_models"), problem_type, len(features_train))
    progress.done("choose_models",
                  f"{len(selected_models)} model(s) selected; scoring on {metric.upper()}")

    # ---------------------------------------------------- 9. training ------
    progress.start("train")

    def on_model(position, total, name):
        progress.update("train", f"Training model {position} of {total}: {name}")

    model_results = training.train_models(
        selected_models, features_train, target_train, features_test, target_test,
        problem_type, metric, progress_callback=on_model)

    trained = [r for r in model_results if r["status"] == "trained"]
    failed = [r for r in model_results if r["status"] == "failed"]
    progress.done("train",
                  f"{len(trained)} model(s) trained successfully"
                  + (f", {len(failed)} failed" if failed else ""))
    for failure in failed:
        warnings_list.append(
            f"{failure['model_name']} failed to train: {failure['error_message']}")

    if not trained:
        progress.fail("evaluate", "Every model failed to train.")
        return {"status": "failed",
                "error": "Every selected model failed to train. The first error was: "
                         + (failed[0]["error_message"] if failed else "unknown"),
                "profile_before": profile_before, "cleaning": cleaning_report,
                "model_results": _strip_unserialisable(model_results)}

    progress.start("evaluate")
    progress.done("evaluate",
                  f"Evaluated on {len(features_test):,} held-out rows using {metric.upper()}")

    # ------------------------------------------------- 10. best model ------
    progress.start("best")
    best = training.pick_best_model(model_results, metric)
    baseline_score = training.get_baseline_score(model_results, metric)
    cross_validation = training.cross_validate_best(
        best, feature_frame.pipe(features.apply_feature_engineering, recipe)[kept],
        target, problem_type, metric)
    progress.done("best",
                  f"{best['model_name']} with {metric.upper()} {best['primary_score']:.4f}")

    importance = training.extract_feature_importance(
        best, features_test, target_test, problem_type)

    # -------------------------------------- 11. did we meet the goal? ------
    progress.start("validate")
    validation = training.validate_objective(
        best, problem_type, metric, baseline_score, target,
        recommendation.get("objective_success_criteria"), cross_validation)
    progress.done("validate", f"Objective {validation['status'].lower()}")

    # ------------------------------------------------- 12. the charts ------
    progress.start("charts")
    chart_files = chart_builder.generate_all(
        analysis_id, chart_dir,
        profile=profile_after,
        target_series=target,
        problem_type=problem_type,
        results=model_results,
        metric=metric,
        best_result=best,
        importance=importance,
        actual=target_test.to_numpy() if problem_type == "regression" else None,
        predicted=best.get("predictions") if problem_type == "regression" else None,
        dataframe=modelling_frame,
    )
    progress.done("charts", f"{len(chart_files)} chart(s) generated")

    # ------------------------------------------------ 13. the write-up -----
    progress.start("report")
    insights = _build_insights(profile_after, cleaning_report, selection, best,
                               validation, importance, problem_type, metric)
    recommendations = _build_recommendations(profile_after, validation, problem_type,
                                             best, metric, selection)
    progress.done("report", "Report ready")

    return {
        "status": "completed",
        "objective": objective,
        "ai_source": ai_source,
        "ai_note": ai_note,
        "recommendation": recommendation,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "cleaning": cleaning_report,
        "problem_type": problem_type,
        "problem_type_reason": type_reason,
        "target_column": target_column,
        "target_confidence": confidence,
        "metric": metric,
        "metric_reason": metric_reason,
        "split_note": split_note,
        "train_rows": int(len(features_train)),
        "test_rows": int(len(features_test)),
        "sampled": sampled,
        "engineering": {"created": recipe["created"], "count": len(recipe["created"])},
        "selection": selection,
        "models_selected": selected_models,
        "models_rejected": rejected_models,
        "model_results": _strip_unserialisable(model_results),
        "best": {k: v for k, v in best.items() if k not in ("pipeline", "predictions")},
        "baseline_score": baseline_score,
        "cross_validation": cross_validation,
        "importance": importance,
        "objective_validation": validation,
        "charts": chart_files,
        "insights": insights,
        "recommendations": recommendations,
        "warnings": warnings_list,
    }


def load_csv(path):
    """
    Read a CSV defensively.

    Real files arrive with the wrong encoding and stray separators far more often
    than anyone expects, so we try the common combinations before giving up.
    """
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin-1", "sep": ";"},
    ]
    last_error = None
    for options in attempts:
        try:
            frame = pd.read_csv(path, low_memory=False, **options)
            if frame.shape[1] == 1 and options.get("sep") is None:
                continue          # probably the wrong separator; try the next option
            if frame.shape[1] >= 1 and frame.shape[0] > 0:
                frame.columns = [str(c).strip() for c in frame.columns]
                return frame
        except Exception as exc:                 # noqa: BLE001
            last_error = exc
    raise ValueError(f"Could not read this CSV file. Last error: {last_error}")
