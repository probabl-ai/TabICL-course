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
# ten, that promise has to generalize to hours the model has never seen.
#
# Models are surprisingly bad at keeping it. A model can be accurate on
# average and still hand out ranges far too narrow, which turns every plan
# built on them into an underestimate. Coverage measures that: the fraction of
# future hours falling inside the promised range.
#
# We put two models through that test. Getting a range out of
# `HistGradientBoostingRegressor` means fitting it once per quantile under the
# pinball loss. TabICL needs no loss and no second fit, because a single
# `predict` call already describes the whole spread of plausible values. We
# draw both ranges over unseen hours to see which one the future falls into,
# repeat the measurement over the ten folds of the same `TimeSeriesSplit` the
# previous notebook used, then hand both models three times the history to see
# what changes.

# %% [markdown]
# ## Learning objectives
#
# - Build an 80 percent prediction band by fitting two quantiles.
# - Read several quantiles out of one TabICL prediction.
# - Overlay both bands on unseen hours and read their coverage off the plot.
# - Tell calibration apart from sharpness, and see why the wider band is not
#   automatically the safer one.
# - Measure coverage across `TimeSeriesSplit` folds instead of trusting one.
# - Explain why a model that fits nothing to our rows is the better
#   calibrated one.

# %% [markdown]
# ## An unbroken stretch of hours to split
#
# We stay with the bike sharing data from the previous notebook. A handful of
# hours are missing, so consecutive rows are not always one hour apart. We
# want them to be, so that "hours into the forecast" means what it says. The
# `hour` column lets us check, since inside an unbroken stretch it increases
# by one and wraps around at midnight.

# %%
import numpy as np
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
).frame

hour = bike_sharing["hour"].to_numpy()
bounds = np.concatenate(([0], np.flatnonzero(np.diff(hour) % 24 != 1) + 1, [len(bike_sharing)]))
longest = np.argmax(np.diff(bounds))
stretch = bike_sharing.iloc[bounds[longest] : bounds[longest] + np.diff(bounds)[longest]]
print(f"longest gap-free stretch: {len(stretch)} consecutive hours")

# %% [markdown]
# Over six months, enough for several train and test pairs. We reuse the
# splitter from the previous notebook unchanged: each fold trains on 800 hours
# and is judged on the 200 that follow, with a 48-hour gap between them so the
# test hours do not start where the training data stops.

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(n_splits=10, gap=48, max_train_size=800, test_size=200)

target_name = "count"
feature_names = [name for name in bike_sharing.columns if name != target_name]
data, target = stretch[feature_names], stretch[target_name]

# %% [markdown]
# ## Two quantiles make a band
#
# The pinball loss generalizes the absolute error, which weighs errors above
# and below the prediction equally and is therefore minimized by the median.
# The pinball loss tilts that balance, so minimizing it at `quantile=0.9`
# gives a value the target stays under about 90 percent of the time.
#
# Fitting it at 0.1 and at 0.9 gives a lower and an upper bound. Between them
# sits a band we expect to catch 80 percent of the test hours. Coverage is
# whether it does. Each bound costs a separate fit, and each fit learns that
# bound alone.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

quantiles = [0.1, 0.9]


def hgbt_band(data_train, target_train, data_test):
    return {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, categorical_features="from_dtype", random_state=0
        )
        .fit(data_train, target_train)
        .predict(data_test)
        for q in quantiles
    }


# %% [markdown]
# ## Both bounds from a single prediction
#
# TabICL does not minimize a loss we choose, and `fit` does not minimize one
# either. It only stores our rows as the context, as the previous notebook
# showed. So there is no quantile to set and no second model to fit. What
# TabICL holds internally is a distribution over plausible values for each
# row, and `predict` decides what to report from it. Its default,
# `output_type="mean"`, collapses that distribution to the single number we
# used before. Asking for `"quantiles"` reports several of its percentiles, so
# one call returns both bounds.
#
# The previous notebook trimmed TabICL's ensemble to 4 members for speed. We
# trim to 1 here, since we call it on twelve folds. Coverage and band width
# barely move when we do.

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor


def tabicl_band(data_train, target_train, data_test):
    fitted = TabICLRegressor(n_estimators=1, random_state=0).fit(data_train, target_train)
    predicted = fitted.predict(data_test, output_type="quantiles", alphas=quantiles)
    return {q: predicted[:, i] for i, q in enumerate(quantiles)}


# %% [markdown]
# ## Measuring every fold
#
# For each fold we build both bands and record how often each one catches the
# target and how wide it had to be. We keep the last fold's bands aside to
# look at one forecast in detail.

# %%
import pandas as pd
from sklearn.metrics import mean_pinball_loss


def evaluate(cv):
    records, last = [], {}
    for fold, (train_index, test_index) in enumerate(cv.split(data)):
        data_train, target_train = data.iloc[train_index], target.iloc[train_index]
        data_test = data.iloc[test_index]
        target_test = target.iloc[test_index].to_numpy()
        bands = {
            "HGBT": hgbt_band(data_train, target_train, data_test),
            "TabICL": tabicl_band(data_train, target_train, data_test),
        }
        for name, band in bands.items():
            inside = (target_test >= band[0.1]) & (target_test <= band[0.9])
            records.append({
                "fold": fold,
                "model": name,
                "coverage": inside.mean(),
                "width": np.mean(band[0.9] - band[0.1]),
                "pinball": np.mean([mean_pinball_loss(target_test, band[q], alpha=q) for q in quantiles]),
            })
        last = {"bands": bands, "target": target_test, "n_train": len(train_index)}
    return pd.DataFrame(records), last


results, last_fold = evaluate(ts_cv)

# %% [markdown]
# ## Where does the future actually land?
#
# We draw both bands over the last fold's 200 test hours, translucent so the
# narrower one stays readable inside the wider one.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
hours_ahead = np.arange(len(last_fold["target"]))

fig, ax = plt.subplots(figsize=(11, 4))
for name, color in [("HGBT", "tab:blue"), ("TabICL", "tab:orange")]:
    band = last_fold["bands"][name]
    ax.fill_between(hours_ahead, band[0.1], band[0.9], alpha=0.5, color=color, label=name)
ax.plot(hours_ahead, last_fold["target"], color="black", linewidth=1, label="actual")
ax.set_xlabel("hours into the forecast")
ax.set_ylabel("bikes rented")
ax.set_title("80% prediction bands on unseen hours")
ax.legend()
plt.show()

# %% [markdown]
# Both bands follow the daily cycle, rising at the commute peaks and
# collapsing overnight. Look at the peaks, though. The black line shoots out
# of the top of the blue band at most of them, while the orange band stretches
# up to meet them and stays tighter in between.
#
# One fold is one stretch of weather and one stretch of the calendar, so it
# settles nothing on its own. The other nine tell us whether what we just saw
# is the rule.

# %%
color = {"whiskers": "black", "medians": "black", "caps": "black"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
for ax, column, label in zip(axes, ["coverage", "width"], ["coverage", "mean band width (bikes)"]):
    results.pivot(index="fold", columns="model", values=column).plot.box(color=color, ax=ax)
    ax.set_ylabel(label)
axes[0].axhline(0.8, color="black", linestyle="--", label="promised 80%")
axes[0].set_ylim(0, 1)
axes[0].legend()
plt.show()

# %%
results.groupby("model")[["coverage", "width", "pinball"]].mean().round(2)

# %% [markdown]
# Averaged over the ten folds, TabICL lands within a point of the 80 percent
# it promised. HGBT catches around 60 percent of the hours, and its box does
# not come close to the dashed line in any fold.

# %%
per_fold = results.pivot(index="fold", columns="model", values=["coverage", "width"])
covers_more = (per_fold["coverage"]["TabICL"] > per_fold["coverage"]["HGBT"]).sum()
narrower = (per_fold["width"]["TabICL"] < per_fold["width"]["HGBT"]).sum()
print(f"TabICL covers more in {covers_more} of {len(per_fold)} folds")
print(f"TabICL is narrower in {narrower} of {len(per_fold)} folds")

# %% [markdown]
# TabICL covers more in every fold, and in seven of the ten it does so with
# the narrower band as well. In the other three it buys the extra coverage by
# widening, the ordinary trade. On average its band is still the tighter one,
# so HGBT ends up wider and less trustworthy at once.
#
# That combination sounds contradictory until you look back at the forecast
# plot. Width is worth nothing unless it sits where the uncertainty is. HGBT
# spreads its band across the ordinary hours and then runs out of room at the
# peaks, exactly where demand varies most. TabICL puts its width where the
# variation is and keeps it tight elsewhere.
#
# This is why coverage and width have to be read together. Any band can buy
# coverage by growing, so coverage alone is a metric you can cheat: a band
# from zero to infinity covers everything and says nothing. Being right as
# often while committing to a narrower range is the gain that cannot be
# faked. The pinball loss in the table above scores both at once, and it ranks
# the two models the same way.

# %% [markdown]
# ## Does more history rescue gradient boosting?
#
# Eight hundred rows is about a month of hourly data. Gradient boosting may
# simply be starved. Raising `max_train_size` to 2400 gives both models three
# times the history on the same kind of folds. Each fold now needs more past
# and costs more to fit, so we ask for two.

# %%
long_cv = TimeSeriesSplit(n_splits=2, gap=48, max_train_size=2400, test_size=200)
long_results, _ = evaluate(long_cv)

comparison = pd.concat(
    [
        results.groupby("model")[["coverage", "width"]].mean().add_suffix(" @800"),
        long_results.groupby("model")[["coverage", "width"]].mean().add_suffix(" @2400"),
    ],
    axis=1,
)
comparison.round(2)

# %% [markdown]
# More history lifts gradient boosting's coverage by more than ten points, but
# three times the data still leaves it short of where TabICL already was on
# 800 rows. TabICL's coverage hardly moves. Its width does: the same promise,
# kept with a narrower band.

# %% [markdown]
# ## Why the pretrained model is the better calibrated one
#
# Both models rest on the pinball loss, so naming the loss explains nothing.
# What differs is how many levels it covers, whether they share anything, and
# which rows it is computed on.
#
# We fit gradient boosting twice, at 0.1 and at 0.9. The two runs share
# nothing: separate trees, separate splits, separate views of how far demand
# can stray. They have separate hyperparameters too, and the learning rate
# that estimates the 10th percentile best is not the one that estimates the
# 90th best. Coverage needs both bounds right at once, which neither fit is
# asked to deliver. What we drew as a band is two error estimates placed side
# by side. No distribution was ever built.
#
# TabICL predicts 999 quantiles, at levels 0.001 through 0.999, in one pass
# and from one representation of the row. Its weights were pretrained by
# summing the pinball loss over all of those levels at once. Knowing a
# target's value at every probability level is the same as knowing its
# distribution, so those 999 numbers are a distribution already. Extend the
# tails with a parametric model and a CDF, a density or any quantile follows.
# Our two bounds are two points on that one curve, and the `"mean"` from the
# previous notebook is the average of the same 999 numbers.

# %% [markdown]
# That explains why TabICL's band is coherent. Why it is also honest comes
# down to which rows the loss is computed on.
#
# Gradient boosting minimizes its pinball loss on the 800 rows it is fitting.
# It can always lower that loss by following those rows more closely, which
# drags the 10th and 90th percentiles toward values it has already seen.
# Narrowing the band is rewarded. Errors it has absorbed cannot warn it about
# errors it has not met, so its promise holds up far better on the training
# rows than on the test rows.
#
# TabICL minimizes nothing here. The same loss was minimized once, during
# pretraining. Each synthetic table was cut into a context and a set of rows
# kept out of it, the model saw only the context, and the loss was scored on
# the rows it had been denied. Narrowing the band is punished there, exactly
# as it is punished on our 200 test hours. Calling `fit` only hands over the
# context and changes none of it.
#
# The two models are not taught different lessons about uncertainty. They are
# graded on different rows. One is scored on rows it has seen, the other was
# scored on rows withheld from it.
#
# A small symptom of the same split shows up in the bounds themselves. Nothing
# ties the two gradient boosting fits together, so on a few quiet overnight
# hours they cross and the band turns inside out by a bike or two.

# %%
for name, band in last_fold["bands"].items():
    crossings = (band[0.1] > band[0.9]).sum()
    print(f"{name:7s} hours where the lower bound sits above the upper one: {crossings}")

# %% [markdown]
# The same gap appears in classification, in a later notebook. There it is
# tempting to credit the log loss for TabICL's better calibrated
# probabilities. Gradient boosting minimizes the log loss too, so naming a
# loss will not be the answer there either.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * fit `HistGradientBoostingRegressor` twice under the pinball loss, at
#   quantiles 0.1 and 0.9, to obtain an 80 percent prediction band;
# * read the same two quantiles out of a single TabICL prediction with
#   `output_type="quantiles"`, with no loss to choose and no second fit;
# * measured coverage, the fraction of hours a band catches, over the ten
#   folds of a `TimeSeriesSplit` rather than on one lucky window;
# * separated calibration from sharpness, and saw that a band can be both
#   wider and less trustworthy;
# * tripled the training window and watched gradient boosting close part of
#   the gap while TabICL converted the extra history into a narrower band
#   instead;
# * traced the gap to two things: TabICL estimates a whole quantile function
#   at once where gradient boosting estimates two unrelated bounds, and its
#   loss was minimized during pretraining on rows held out of the context,
#   not on the rows being fitted.
