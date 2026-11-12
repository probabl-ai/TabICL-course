# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Tabular foundation models predict without training on your data
#
# The models we built in the
# [Machine learning in Python with scikit-learn](https://skolar.probabl.ai/)
# course, such as linear models or gradient-boosted trees, learn by adjusting
# their parameters to the training data. A pretrained large language model
# works differently. We don't retrain it for a new task. Instead, we show it a
# few examples in a prompt and it answers.
#
# Tabular foundation models, or TFMs, bring that idea to tables. TabICL is one
# of them. Its transformer was pretrained once, on millions of synthetic
# tables, so that it can read our training rows as context and predict new
# rows right away.
#
# In this notebook, we compare TabICL with a `HistGradientBoostingRegressor`
# (HGBT) on the bike sharing dataset, first with default hyperparameters, then
# giving HGBT a growing tuning budget.

# %% [markdown]
# ## The bike sharing dataset
#
# Each row is one hour of Capital Bikeshare, Washington D.C.'s bike-sharing
# system, in 2011 or 2012. The columns describe the calendar and the weather.
# `count`, our target, is the number of bikes rented that hour. The
# [bike sharing dataset notebook](datasets_bike_sharing.ipynb) explores the
# data and explains how we prepared the CSV file we load here.

# %%
import pandas as pd

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")

target_name = "count"
target = bike_sharing[target_name]
data = bike_sharing.drop(columns=target_name)

# %% [markdown]
# Consecutive rows are only an hour apart and strongly correlated. Passing
# `cv=5` would use a `KFold` that doesn't shuffle, so each test fold is a
# contiguous block of hours. The training rows still surround that block, so
# every fold but the last would train on the future, and on the hours right
# next to the ones it is tested on. `TimeSeriesSplit` always trains on the past
# and tests on the future, and a 48-hour `gap` keeps the test set from starting
# right where the training data ends.
#
# Reading a longer context costs TabICL more compute, so we cap the training
# window at 800 rows per fold. Both models share this splitter, so they are
# compared on the same folds.

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(n_splits=10, gap=48, max_train_size=800, test_size=200)

# %% [markdown]
# ## What `fit` does for TabICL
#
# The `tabicl` package wraps the pretrained model behind the usual scikit-learn
# `fit`/`predict` interface. Its `fit` doesn't train anything. Instead, it loads
# the pretrained weights and stores our training rows as context. The first call
# also downloads the weights, about 110 MB, to a local cache.

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor

tabicl = TabICLRegressor(random_state=0)

# %%
# %%time
_ = tabicl.fit(data.iloc[:100], target.iloc[:100])

# %% [markdown]
# Let's call `fit` a second time:

# %%
# %%time
_ = tabicl.fit(data.iloc[:100], target.iloc[:100])

# %% [markdown]
# This time the weights are already on disk, and reading them back takes a
# fraction of a second. The expensive part happens in `predict`, where the
# transformer reads the stored rows to produce each prediction.

# %% [markdown]
# ## Comparing both models with default hyperparameters
#
# HGBT handles the `category` columns natively, so neither model needs any
# preprocessing.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate

cv_results_hgbt = cross_validate(
    HistGradientBoostingRegressor(random_state=0),
    data,
    target,
    cv=ts_cv,
    scoring="neg_mean_absolute_error",
)
cv_results_tabicl = cross_validate(
    TabICLRegressor(random_state=0),
    data,
    target,
    cv=ts_cv,
    scoring="neg_mean_absolute_error",
)

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

cv_results = pd.DataFrame(
    {"HGBT": -cv_results_hgbt["test_score"], "TabICL": -cv_results_tabicl["test_score"]}
)

color = {"whiskers": "black", "medians": "black", "caps": "black"}
cv_results.plot.box(color=color, vert=False)
plt.xlabel("Mean absolute error (bikes per hour)")
plt.title("Test MAE across folds (800 training rows, 200 test rows)")
plt.show()

# %% [markdown]
# TabICL's errors sit lower, around 37 bikes per hour on average against 42 for
# HGBT. Both boxes are wide, because some folds land in an unusually quiet or
# busy stretch that 800 rows of history don't anticipate.

# %% [markdown]
# ## Error versus computation time
#
# `cross_validate` also records the duration of `fit` and `predict` on each
# fold. We summarize each model as a single point, namely its mean MAE and mean
# time across folds, and add error bars for the standard deviation.

# %%
model_results = {"HGBT": cv_results_hgbt, "TabICL": cv_results_tabicl}
model_colors = {"HGBT": "tab:blue", "TabICL": "tab:orange"}

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 10))
for ax, time_key, ylabel in zip(
    axes, ["fit_time", "score_time"], ["Fit time (s)", "Predict time (s)"]
):
    for name, results in model_results.items():
        mae = -results["test_score"]
        times = results[time_key]
        ax.errorbar(
            mae.mean(),
            times.mean(),
            xerr=mae.std(),
            yerr=times.std(),
            fmt="o",
            color=model_colors[name],
            label=name,
        )
    ax.set_ylabel(ylabel)

axes[1].set_xlim(left=0)
axes[1].set_xlabel("Mean absolute error (bikes per hour)")
axes[0].set_title("Fit time vs error")
axes[1].set_title("Predict time vs error")
axes[1].legend()
plt.show()

# %% [markdown]
# Both models fit in under a second, HGBT growing its trees and
# TabICL reading its weights from disk. HGBT predicts 200 rows in hundredths of
# a second, whereas TabICL needs several seconds, because its transformer
# processes the 800 context rows together with the rows to predict.

# %% [markdown]
# ## Tuning HGBT's hyperparameters
#
# HGBT's defaults are not necessarily its best. A `RandomizedSearchCV` lets us
# give it a budget, the number of candidates `n_iter` it may try. More
# candidates should lower the error, up to a point, but each one costs fit
# time. The search needs its own leak-free splitter, so we nest a smaller
# `TimeSeriesSplit` inside each training fold.
#
# We leave `learning_rate` and `max_iter` at their defaults. A smaller learning
# rate needs more trees, so tuning both at once mostly wastes budget. Instead,
# we search over the parameters that control the complexity of each tree, in
# ranges suited to 800 training rows.
#
# Fixing `random_state` makes the searches nested, meaning that the 4 candidates
# tried with `n_iter=4` are the first 4 of the 8 tried with `n_iter=8`, and so
# on. We keep `n_jobs` at its default so the timings stay comparable with
# TabICL's.
#
# ```{tip}
# The full range of budgets takes a couple of minutes to run. On a slow
# machine, start with `budgets = [1]` and add larger budgets one at a time, up
# to `n_iter=16`.
# ```

# %%
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


param_distributions = {
    "max_leaf_nodes": loguniform_int(4, 64),
    "min_samples_leaf": loguniform_int(5, 100),
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": [0.5, 0.8, 1.0],
}
inner_cv = TimeSeriesSplit(n_splits=3, gap=48)

budgets = [1, 4, 8, 16]
# budgets = [1]
cv_results_random_search = {}
for n_iter in budgets:
    random_search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=0),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        random_state=42,
    )
    cv_results_random_search[n_iter] = cross_validate(
        random_search,
        data,
        target,
        cv=ts_cv,
        scoring="neg_mean_absolute_error",
    )


# %%
def plot_error_vs_time(time_keys, ylabel, title, annotate=True):
    _, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for n_iter, results in cv_results_random_search.items():
        mae = -results["test_score"]
        times = sum(results[key] for key in time_keys)
        ax.errorbar(
            mae.mean(),
            times.mean(),
            xerr=mae.std(),
            yerr=times.std(),
            fmt="o",
            color=model_colors["HGBT"],
            label="HGBT (random search)" if n_iter == budgets[0] else None,
        )
        if annotate:
            ax.annotate(
                f"n_iter={n_iter}",
                (mae.mean(), times.mean()),
                textcoords="offset points",
                xytext=(6, 6),
                color=model_colors["HGBT"],
            )

    mae = -cv_results_tabicl["test_score"]
    times = sum(cv_results_tabicl[key] for key in time_keys)
    ax.errorbar(
        mae.mean(),
        times.mean(),
        xerr=mae.std(),
        yerr=times.std(),
        fmt="o",
        color=model_colors["TabICL"],
        label="TabICL",
    )

    ax.set_xlim(left=0)
    ax.set_xlabel("Mean absolute error (bikes per hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()


# %%
plot_error_vs_time(["fit_time"], "Fit time (s)", "Fit time vs error")

# %%
plot_error_vs_time(
    ["score_time"], "Predict time (s)", "Predict time vs error", annotate=False
)

# %%
plot_error_vs_time(
    ["fit_time", "score_time"],
    "Compute time, fit + predict (s)",
    "Compute time vs error",
)

# %% [markdown]
# A single random candidate does much worse than the defaults, and it takes 8
# candidates to match them. Doubling the budget to 16 does not improve the
# error further, which stays at about 42 bikes per hour, above TabICL's 37,
# while fit time more than doubles. At `n_iter=8`, HGBT already spends about
# as much total compute as TabICL, and at `n_iter=16` more than twice as much,
# and it still predicts worse.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * saw that TabICL's `fit` only loads pretrained weights and stores the
#   training rows, downloading the weights on the first call;
# * cross-validated HGBT and TabICL with default hyperparameters on 800-row
#   windows of the bike sharing data, respecting the time order;
# * compared where each model spends its time, in `fit` for HGBT and in
#   `predict` for TabICL;
# * gave HGBT a random search with a growing budget, which did not bring its
#   error below its defaults by much while its fit time kept growing.
#
# The next notebook checks whether TabICL keeps its lead as the training window
# grows.
