"""
Chart generation.

Charts are rendered server-side with matplotlib and saved as PNGs under
static/charts/, so the browser only ever loads an image. Every chart here is a
single-series chart with a named title, which is why none of them carry a
legend: the title says what the bars are.

The palette is fixed and shared with the site's CSS so the dashboard reads as
one design rather than a page with pictures pasted into it.
"""

import os

import matplotlib

matplotlib.use("Agg")           # no display server on a web host

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette ---

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
NEUTRAL = "#c3c2b7"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Single-hue sequential ramp (blue 100 -> 700) for magnitude encoding.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# Diverging blue <-> red with a neutral gray midpoint, for correlations.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#184f95", "#6da7ec", "#f0efec", "#e88a89", "#d03b3b"],
)


def _new_figure(width, height):
    figure, axes = plt.subplots(figsize=(width, height), dpi=110)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(1)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    return figure, axes


def _finish(figure, axes, path, title, subtitle=None):
    axes.set_title(title, color=INK_PRIMARY, fontsize=12.5, fontweight="600",
                   loc="left", pad=24 if subtitle else 10)
    if subtitle:
        # Offset in points, not axes fractions, so the gap stays the same on a
        # short chart and on a tall one.
        axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                      xytext=(0, 7), textcoords="offset points",
                      color=INK_SECONDARY, fontsize=9.5, va="bottom", ha="left")
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return os.path.basename(path)


def _shorten(label, limit=26):
    label = str(label)
    return label if len(label) <= limit else label[: limit - 1] + "…"


# --------------------------------------------------------------------------
# 1. Missing values
# --------------------------------------------------------------------------

def missing_values_chart(profile, output_path):
    columns = [c for c in profile["columns"] if c["missing"] > 0]
    if not columns:
        return None

    columns = sorted(columns, key=lambda c: c["missing_pct"], reverse=True)[:15]
    names = [_shorten(c["name"]) for c in columns][::-1]
    values = [c["missing_pct"] for c in columns][::-1]

    figure, axes = _new_figure(7.2, max(2.4, 0.42 * len(names) + 1.2))
    positions = np.arange(len(names))
    axes.barh(positions, values, height=0.62, color=SERIES_BLUE)
    axes.set_yticks(positions)
    axes.set_yticklabels(names, color=INK_SECONDARY, fontsize=9.5)
    axes.set_xlabel("% of rows missing", color=INK_MUTED, fontsize=9.5)
    axes.xaxis.grid(True, color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.set_xlim(0, max(values) * 1.18)

    for position, value in zip(positions, values):
        axes.text(value + max(values) * 0.02, position, f"{value:.1f}%",
                  va="center", fontsize=9, color=INK_SECONDARY)

    return _finish(figure, axes, output_path, "Missing values by column",
                   f"{len(columns)} column(s) with gaps, worst first")


# --------------------------------------------------------------------------
# 2. Target distribution
# --------------------------------------------------------------------------

def target_distribution_chart(target_series, problem_type, output_path):
    figure, axes = _new_figure(7.2, 3.6)

    if problem_type == "classification":
        counts = target_series.value_counts().head(12)
        labels = [_shorten(index, 18) for index in counts.index]
        positions = np.arange(len(counts))
        axes.bar(positions, counts.to_numpy(), width=0.6, color=SERIES_BLUE)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, color=INK_SECONDARY, fontsize=9.5,
                             rotation=30 if max(len(l) for l in labels) > 8 else 0,
                             ha="right" if max(len(l) for l in labels) > 8 else "center")
        axes.yaxis.grid(True, color=GRID, linewidth=1)
        axes.set_axisbelow(True)
        axes.set_ylabel("rows", color=INK_MUTED, fontsize=9.5)

        total = counts.sum()
        for position, value in zip(positions, counts.to_numpy()):
            axes.text(position, value + total * 0.015, f"{value:,}\n{value / total:.0%}",
                      ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
        axes.set_ylim(0, counts.max() * 1.24)

        share = counts.min() / total
        subtitle = (f"Smallest class is {share:.1%} of rows - "
                    f"{'imbalanced' if share < 0.35 else 'reasonably balanced'}")
    else:
        values = pd.to_numeric(target_series, errors="coerce").dropna()
        axes.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))),
                  color=SERIES_BLUE, edgecolor=SURFACE, linewidth=1)
        axes.yaxis.grid(True, color=GRID, linewidth=1)
        axes.set_axisbelow(True)
        axes.set_xlabel(str(target_series.name), color=INK_MUTED, fontsize=9.5)
        axes.set_ylabel("rows", color=INK_MUTED, fontsize=9.5)
        axes.axvline(values.median(), color=SERIES_ORANGE, linewidth=2)
        axes.text(values.median(), axes.get_ylim()[1] * 0.94, f"  median {values.median():,.2f}",
                  color=SERIES_ORANGE, fontsize=9, va="top")
        subtitle = (f"mean {values.mean():,.2f} · std {values.std():,.2f} · "
                    f"skew {values.skew():.2f}")

    return _finish(figure, axes, output_path,
                   f"Distribution of {target_series.name}", subtitle)


# --------------------------------------------------------------------------
# 3. Model comparison
# --------------------------------------------------------------------------

def model_comparison_chart(results, metric, output_path):
    usable = [r for r in results if r["status"] == "trained" and r["primary_score"] is not None]
    if not usable:
        return None

    lower_is_better = metric in {"rmse", "mae", "mse"}
    usable = sorted(usable, key=lambda r: r["primary_score"], reverse=not lower_is_better)

    names = [_shorten(r["model_name"], 30) for r in usable][::-1]
    scores = [r["primary_score"] for r in usable][::-1]
    # Colour separates *what a bar is* (a real model vs the naive baseline),
    # never how it ranked.
    colours = [NEUTRAL if r["is_baseline"] else SERIES_BLUE for r in usable][::-1]

    figure, axes = _new_figure(7.4, max(2.6, 0.46 * len(names) + 1.3))
    positions = np.arange(len(names))
    axes.barh(positions, scores, height=0.64, color=colours)
    axes.set_yticks(positions)
    axes.set_yticklabels(names, color=INK_SECONDARY, fontsize=9.5)
    axes.set_xlabel(metric.upper(), color=INK_MUTED, fontsize=9.5)
    axes.xaxis.grid(True, color=GRID, linewidth=1)
    axes.set_axisbelow(True)

    span = max(scores) - min(min(scores), 0)
    axes.set_xlim(min(min(scores), 0), max(scores) + span * 0.22 + 1e-9)
    for position, value in zip(positions, scores):
        axes.text(value + span * 0.02, position, f"{value:.4f}",
                  va="center", fontsize=9, color=INK_SECONDARY)

    return _finish(figure, axes, output_path, f"Model comparison by {metric.upper()}",
                   "Grey bar is the naive baseline every real model has to beat")


# --------------------------------------------------------------------------
# 4. Confusion matrix
# --------------------------------------------------------------------------

def confusion_matrix_chart(matrix, classes, output_path):
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return None

    size = max(3.4, min(8.0, 1.0 + 0.75 * len(classes)))
    figure, axes = _new_figure(size + 1.2, size)

    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, np.where(row_totals == 0, 1, row_totals))

    image = axes.imshow(shares, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    axes.set_xticks(np.arange(len(classes)))
    axes.set_yticks(np.arange(len(classes)))
    axes.set_xticklabels([_shorten(c, 14) for c in classes], color=INK_SECONDARY, fontsize=9)
    axes.set_yticklabels([_shorten(c, 14) for c in classes], color=INK_SECONDARY, fontsize=9)
    axes.set_xlabel("predicted", color=INK_MUTED, fontsize=9.5)
    axes.set_ylabel("actual", color=INK_MUTED, fontsize=9.5)

    # 2px surface gap between cells.
    axes.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="minor", length=0)

    for row in range(len(classes)):
        for column in range(len(classes)):
            share = shares[row, column]
            axes.text(column, row, f"{matrix[row, column]:,}\n{share:.0%}",
                      ha="center", va="center", fontsize=9,
                      color="#ffffff" if share > 0.55 else INK_PRIMARY)

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar.set_label("share of each actual class", color=INK_MUTED, fontsize=8.5)

    return _finish(figure, axes, output_path, "Confusion matrix",
                   "Each row is an actual class; the diagonal is what the model got right")


# --------------------------------------------------------------------------
# 5. Feature importance
# --------------------------------------------------------------------------

def feature_importance_chart(importance, output_path):
    features = importance.get("features") or []
    if not features:
        return None

    features = features[:15][::-1]
    names = [_shorten(f["feature"], 30) for f in features]
    values = [f["importance"] for f in features]

    figure, axes = _new_figure(7.2, max(2.6, 0.44 * len(names) + 1.3))
    positions = np.arange(len(names))
    axes.barh(positions, values, height=0.64, color=SERIES_BLUE)
    axes.set_yticks(positions)
    axes.set_yticklabels(names, color=INK_SECONDARY, fontsize=9.5)
    axes.xaxis.grid(True, color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.set_xlabel("importance", color=INK_MUTED, fontsize=9.5)

    span = max(values) if values else 1
    axes.set_xlim(0, span * 1.2)
    for position, feature in zip(positions, features):
        axes.text(feature["importance"] + span * 0.02, position, f"{feature['share']:.1f}%",
                  va="center", fontsize=9, color=INK_SECONDARY)

    return _finish(figure, axes, output_path, "What the winning model relied on",
                   importance.get("method", ""))


# --------------------------------------------------------------------------
# 6. Actual vs predicted (regression)
# --------------------------------------------------------------------------

def actual_vs_predicted_chart(actual, predicted, output_path):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return None

    if actual.size > 3000:                       # keep the PNG readable and small
        sample = np.random.default_rng(42).choice(actual.size, 3000, replace=False)
        actual, predicted = actual[sample], predicted[sample]

    figure, axes = _new_figure(6.4, 5.2)
    axes.scatter(actual, predicted, s=26, color=SERIES_BLUE, alpha=0.55,
                 edgecolors=SURFACE, linewidths=2)

    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    axes.plot([low, high], [low, high], color=INK_MUTED, linewidth=2, linestyle="--")
    axes.text(high, high, "  perfect prediction", color=INK_MUTED, fontsize=9,
              ha="right", va="bottom")

    axes.set_xlabel("actual", color=INK_MUTED, fontsize=9.5)
    axes.set_ylabel("predicted", color=INK_MUTED, fontsize=9.5)
    axes.grid(True, color=GRID, linewidth=1)
    axes.set_axisbelow(True)

    return _finish(figure, axes, output_path, "Predicted against actual values",
                   "Points near the dashed line are accurate predictions")


# --------------------------------------------------------------------------
# 7. Correlation heatmap
# --------------------------------------------------------------------------

def correlation_chart(dataframe, output_path, max_columns=12):
    numeric = dataframe.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None

    if numeric.shape[1] > max_columns:
        variance_ranked = numeric.var(numeric_only=True).sort_values(ascending=False)
        numeric = numeric[variance_ranked.index[:max_columns]]

    matrix = numeric.corr()
    size = max(4.0, min(9.0, 0.62 * matrix.shape[0] + 2.0))
    figure, axes = _new_figure(size + 1.0, size)

    image = axes.imshow(matrix.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [_shorten(c, 16) for c in matrix.columns]
    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right", color=INK_SECONDARY, fontsize=8.5)
    axes.set_yticklabels(labels, color=INK_SECONDARY, fontsize=8.5)

    axes.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="minor", length=0)

    if len(labels) <= 10:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix.iloc[row, column]
                axes.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=8, color="#ffffff" if abs(value) > 0.62 else INK_PRIMARY)

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    return _finish(figure, axes, output_path, "How the numeric columns move together",
                   "Blue is a negative relationship, red positive, grey none")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_all(analysis_id, chart_dir, profile=None, target_series=None, problem_type=None,
                 results=None, metric=None, best_result=None, importance=None,
                 actual=None, predicted=None, dataframe=None):
    """
    Render every chart that makes sense for this analysis.

    Anything that cannot be drawn (no missing values, a classification problem
    with no regression scatter) is simply absent from the returned dictionary
    rather than an empty placeholder.
    """
    charts = {}

    def path_for(name):
        return os.path.join(chart_dir, f"{analysis_id}_{name}.png")

    def attempt(key, function, *args):
        try:
            filename = function(*args, path_for(key))
            if filename:
                charts[key] = filename
        except Exception:                        # noqa: BLE001 - a chart is never fatal
            pass

    if profile is not None:
        attempt("missing", missing_values_chart, profile)
    if target_series is not None and problem_type in ("classification", "regression"):
        attempt("target", target_distribution_chart, target_series, problem_type)
    if dataframe is not None:
        attempt("correlation", correlation_chart, dataframe)
    if results and metric:
        attempt("comparison", model_comparison_chart, results, metric)
    if best_result and problem_type == "classification":
        matrix = best_result.get("metrics", {}).get("confusion_matrix")
        classes = best_result.get("metrics", {}).get("classes")
        if matrix and classes:
            attempt("confusion", confusion_matrix_chart, matrix, classes)
    if importance:
        attempt("importance", feature_importance_chart, importance)
    if problem_type == "regression" and actual is not None and predicted is not None:
        attempt("scatter", actual_vs_predicted_chart, actual, predicted)

    return charts
