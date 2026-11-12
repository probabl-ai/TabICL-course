# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Does your model know what it does not know?
#
# In the previous notebook we asked which model predicts the right number of
# bikes. But forecasting exactly 180 bikes at 6pm is too confident a claim to
# plan around. A range is something an operator can better act on. If a model
# promises that demand will land between 40 and 300 bikes eight times out of
# ten, that range is an 80 percent prediction interval (PI). Its promise has
# to generalize to hours the model has never seen.
#
# Models are surprisingly bad at keeping it. A model can be accurate on
# average and still hand out PIs far too narrow, which turns every plan
# built on them into an underestimate. Coverage measures this failure as the
# fraction of future hours falling inside the PI.
#
# We put two models through that test. Getting a PI out of
# `HistGradientBoostingRegressor` means fitting it once per quantile under the
# pinball loss. TabICL needs no loss and no second fit, because a single
# `predict` call already describes the whole spread of plausible values. We
# draw both PIs over unseen hours to see which one the future falls into,
# repeat the measurement over the ten folds of the same `TimeSeriesSplit` the
# previous notebook used, then hand both models three times the history to see
# what changes.

# %% [markdown]
# ## Learning objectives
#
# - Build an 80 percent PI by fitting two quantiles.
# - Read several quantiles out of one TabICL prediction.
# - Overlay both PIs on unseen hours and read their coverage off the plot.
# - Tell calibration apart from sharpness, and see why the wider PI is not
#   automatically the safer one.
# - Measure coverage across `TimeSeriesSplit` folds instead of trusting one.
# - Explain why a model that fits nothing to our rows has the better
#   calibrated PI.

# %% [markdown]
# ## Data and time-based splits
#
# We stay with the bike sharing data and the splitter from the previous
# notebooks. The data stops at the end of its longest gap-free stretch (see
# the [bike sharing dataset notebook](datasets_bike_sharing.ipynb)), so every
# test window spans consecutive hours and "hours into the forecast" means what
# it says. Each fold trains on 800 hours and is
# judged on the 200 that follow, with a 48-hour gap between them so the test
# hours do not start where the training data stops.

# %%
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")

target_name = "count"
target = bike_sharing[target_name]
data = bike_sharing.drop(columns=target_name)

ts_cv = TimeSeriesSplit(n_splits=10, gap=48, max_train_size=800, test_size=200)

# %% [markdown]
# ## Prediction intervals from quantiles
#
# The pinball loss generalizes the absolute error, which weighs errors above
# and below the prediction equally and is therefore minimized by the median.
# The pinball loss tilts that balance, so minimizing it at `quantile=0.9`
# gives a value the target stays under about 90 percent of the time.
#
# Fitting it at 0.1 and at 0.9 gives a lower and an upper bound. Between them
# sits a PI we expect to catch 80 percent of the test hours, and coverage
# tells us whether it does. Each bound costs a separate fit.
#
# We wrap the two fits in a small estimator whose `predict` returns both
# bounds as columns. Scikit-learn's cross-validation tools can then treat the
# PI as a single model.

# %%
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor

quantiles = [0.1, 0.9]


class HGBTInterval(BaseEstimator):
    def fit(self, X, y):
        self.models_ = [
            HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features="from_dtype", random_state=0
            ).fit(X, y)
            for q in quantiles
        ]
        return self

    def predict(self, X):
        return np.column_stack([model.predict(X) for model in self.models_])


# %% [markdown]
# TabICL has no loss for us to choose, and its `fit` minimizes nothing. As the
# previous notebook showed, it only stores our rows as the context, so there
# is no quantile to set and no second model to fit. What
# TabICL holds internally is a distribution over plausible values for each
# row, and `predict` decides what to report from it. Its default,
# `output_type="mean"`, collapses that distribution to the single number we
# used before. Asking for `"quantiles"` reports several of its percentiles, so
# one call returns both bounds. We wrap it the same way as gradient boosting.
#
# We fit TabICL on many folds, so we use a single ensemble member,
# `n_estimators=1`.

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor


class TabICLInterval(BaseEstimator):
    def fit(self, X, y):
        self.model_ = TabICLRegressor(n_estimators=1, random_state=0).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X, output_type="quantiles", alphas=quantiles)


models = {"HGBT": HGBTInterval(), "TabICL": TabICLInterval()}

# %% [markdown]
# ## Coverage and width of the PIs
#
# We start with a single forecast. Both models are fit on the last fold's 800
# training hours, and we draw their PIs over the 200 test hours that follow,
# translucent so the narrower one stays readable inside the wider one.

# %%
import matplotlib.pyplot as plt
from sklearn.base import clone

train_index, test_index = list(ts_cv.split(data))[-1]
target_test = target.iloc[test_index].to_numpy()
intervals = {
    name: clone(model)
    .fit(data.iloc[train_index], target.iloc[train_index])
    .predict(data.iloc[test_index])
    for name, model in models.items()
}

plt.rcParams["figure.dpi"] = 200
hours_ahead = np.arange(len(test_index))

fig, ax = plt.subplots(figsize=(11, 4))
for name, color in [("HGBT", "tab:blue"), ("TabICL", "tab:orange")]:
    ax.fill_between(
        hours_ahead,
        intervals[name][:, 0],
        intervals[name][:, 1],
        alpha=0.5,
        color=color,
        label=name,
    )
ax.plot(hours_ahead, target_test, color="black", linewidth=1, label="actual")
ax.set_xlabel("hours into the forecast")
ax.set_ylabel("bikes rented")
ax.set_title("80% prediction intervals on unseen hours")
ax.legend()
plt.show()

# %% [markdown]
# Both PIs follow the daily cycle, rising at the commute peaks and
# collapsing overnight. Look at the peaks, though. The black line shoots out
# of the top of the blue PI at most of them, while the orange PI stretches
# up to meet them and stays tighter in between.
#
# One fold is one stretch of weather and one stretch of the calendar, so it
# settles nothing on its own. `cross_validate` repeats the measurement over
# all ten folds. It scores a model by calling its `predict`, so `make_scorer`
# turns coverage, mean width and the average pinball loss of the two bounds
# into scorers that read the two columns of the PI.

# %%
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import cross_validate


def coverage(y_true, interval):
    return np.mean((y_true >= interval[:, 0]) & (y_true <= interval[:, 1]))


def width(y_true, interval):
    return np.mean(interval[:, 1] - interval[:, 0])


def pinball(y_true, interval):
    return np.mean(
        [mean_pinball_loss(y_true, interval[:, i], alpha=q) for i, q in enumerate(quantiles)]
    )


scoring = {
    "coverage": make_scorer(coverage),
    "width": make_scorer(width),
    "pinball": make_scorer(pinball),
}
metrics = ["test_coverage", "test_width", "test_pinball"]


def evaluate(cv):
    return pd.concat(
        {
            name: pd.DataFrame(cross_validate(model, data, target, cv=cv, scoring=scoring))
            for name, model in models.items()
        },
        names=["model", "fold"],
    )[metrics]


results = evaluate(ts_cv)

# %%
color = {"whiskers": "black", "medians": "black", "caps": "black"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
for ax, column, label in zip(axes, metrics, ["coverage", "mean PI width (bikes)"]):
    results[column].unstack("model").plot.box(color=color, ax=ax)
    ax.set_ylabel(label)
axes[0].axhline(0.8, color="black", linestyle="--", label="promised 80%")
axes[0].set_ylim(0, 1)
axes[0].legend()
plt.show()

# %%
results.groupby("model").agg(["mean", "std"]).round(2)

# %% [markdown]
# Averaged over the ten folds, TabICL lands within a point of the 80 percent
# it promised. HGBT catches around 60 percent of the hours, and its box does
# not come close to the dashed line. TabICL's PI is also the narrower one on
# average, so HGBT ends up wider and less trustworthy at once.
#
# Coverage and width have to be read together, because any PI can raise its
# coverage by growing. A PI from zero to infinity contains every hour and
# tells the operator nothing useful. The pinball loss in the table above
# scores coverage and width at once, and it ranks the two models the same way.
#
# Eight hundred rows is about a month of hourly data. Gradient boosting may
# simply be starved. Raising `max_train_size` to 2400 gives both models three
# times the history. TabICL costs more to run on a longer context, so we keep
# only two folds. `TimeSeriesSplit` always places its test windows at the end
# of the data, so these two folds are tested on the same 400 hours as the last
# two folds above, and we compare against those.

# %%
long_cv = TimeSeriesSplit(n_splits=2, gap=48, max_train_size=2400, test_size=200)
long_results = evaluate(long_cv)

pd.concat(
    {
        "800 rows": results.groupby("model").tail(2).groupby("model").mean(),
        "2400 rows": long_results.groupby("model").mean(),
    },
    axis=1,
).round(2)

# %% [markdown]
# On these two folds, more history lifts gradient boosting's coverage by more
# than ten points, partly by widening its PI. Three times the data still
# leaves it short of where TabICL already was on 800 rows. TabICL's coverage
# and width both hardly move.

# %% [markdown]
# ## Why TabICL's PI is better calibrated
#
# For a regression model, calibration means that the PIs keep their promised
# coverage. An 80 percent PI should catch about 80 percent of unseen hours.
# TabICL's does, and gradient boosting's does not.
#
# Both models are trained with the pinball loss, which rewards quantiles that
# match the distribution of the true values. What differs is the rows the
# loss is computed on.
#
# Gradient boosting minimizes it on its 800 training rows. The trees have
# already fit those rows, so their errors there are smaller than on hours they
# have never seen. Each bound is placed for these smaller training errors, so
# the PI comes close to its promise on the training rows and breaks on the
# test hours.
#
# TabICL's pinball loss was minimized during pretraining, on rows held out of
# the context of each synthetic table. On held-out rows, the errors have two
# sources, the noise in the target and the model's uncertainty about the
# underlying relationship, since all it knows is the context. TabICL's PIs are
# sized for both, which is exactly the situation of our test hours. Our 800
# rows only serve as context.
#
# Gradient boosting could also be scored on held-out rows, for instance by
# tuning its hyperparameters on validation folds with the pinball loss. These
# folds would come out of the same 800 rows, which the learning curve already
# showed to be few for gradient boosting. TabICL needs no validation data,
# because its held-out scoring happened before it saw our table. The same
# argument returns in a later notebook, for the probabilities of a
# classifier.
#
# The two models also build their PIs differently. Gradient boosting fits
# each bound as a separate model, with its own trees, and nothing keeps the
# lower bound below the upper one. On a few quiet overnight hours, the PI
# turns inside out by a bike or two.

# %%
for name, interval in intervals.items():
    crossings = (interval[:, 0] > interval[:, 1]).sum()
    print(f"{name:7s} hours where the lower bound sits above the upper one: {crossings}")

# %% [markdown]
# TabICL predicts 999 quantiles, at levels 0.001 through 0.999, in one pass.
# Together they describe one distribution of plausible values for each row,
# and our two bounds are two points on it, so they cannot cross.
#
# Calibration is about matching the PI to the errors, whatever their size.
# TabICL's PI can be narrower and still keep its promise because its
# forecasts sit closer to the actual counts, as the previous notebook showed,
# so a narrower range around them is enough to catch 80 percent of the hours.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * fit `HistGradientBoostingRegressor` twice under the pinball loss, at
#   quantiles 0.1 and 0.9, to obtain an 80 percent PI;
# * read the same two quantiles out of a single TabICL prediction with
#   `output_type="quantiles"`, with no loss to choose and no second fit;
# * measured coverage, the fraction of hours a PI catches, over the ten
#   folds of a `TimeSeriesSplit` rather than on one lucky window;
# * separated calibration from sharpness, and saw that a PI can be both
#   wider and less trustworthy;
# * tripled the training window and watched gradient boosting close part of
#   the gap while TabICL's PI stayed about the same;
# * traced the gap in coverage to the rows each loss is computed on, since
#   gradient boosting fits its bounds on its training rows while TabICL was
#   pretrained on rows held out of the context, and needs no validation data;
# * saw that TabICL's two bounds come from one distribution, while gradient
#   boosting's come from two separate fits that can cross.
