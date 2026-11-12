# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # When does caching the training rows pay off?
#
# In the learning curve notebook, TabICL's time per fold grew with the
# training window. TabICL learns in context, so at each call to `predict`, the
# transformer reads the training rows and the rows to predict together. Part
# of that work depends only on the training rows. Each layer of the
# transformer turns every training row into intermediate results, which the
# rows to predict then look up. These results stay the same whatever rows we
# predict.
#
# Setting `kv_cache=True` computes them once, during `fit`, and stores them.
# Every later call to `predict` then only processes the new rows. The name of
# the parameter comes from these stored results, which a transformer calls
# keys and values.
#
# In this exercise, we:
#
# * check that the cache leaves the predictions unchanged;
# * measure how it moves compute time between `fit` and `predict`;
# * work out when the cache saves time overall.

# %% [markdown]
# We reuse the bike sharing data and the time-aware splitter of the intro
# notebook, namely ten folds of 800 training hours and 200 test hours, with a
# 48-hour gap between them.

# %%
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")
target = bike_sharing["count"]
data = bike_sharing.drop(columns="count")

ts_cv = TimeSeriesSplit(n_splits=10, gap=48, max_train_size=800, test_size=200)

# %% [markdown]
# As in the previous exercise, the next cell runs TabICL once on a small subset
# so that one-off costs, such as reading the weights from disk for the first
# time, don't inflate the timings of whichever setting runs first.

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor

_ = (
    TabICLRegressor(n_estimators=1, random_state=0)
    .fit(data.iloc[:100], target.iloc[:100])
    .predict(data.iloc[100:120])
)

# %% [markdown]
# ## Does the cache change the predictions?
#
# The cache should only change when the work happens, not its result. Before
# looking at timings, we check that claim on our data.
#
# With 800 training rows, the previous exercise found no measurable gain from
# extra ensemble members, so we use a single one. Cross-validate a
# `TabICLRegressor` with `n_estimators=1` and `random_state=0`, once with
# `kv_cache=False` and once with `kv_cache=True`, using `ts_cv` and the mean
# absolute error as scoring. Collect the results in a dataframe named
# `cv_results`, with one row per setting and fold. Its columns should be
# `kv_cache`, `fold`, `mae`, `fit_time`, `score_time`, and `total_time`, the
# sum of the two times. Then display the mean and the standard deviation of
# the MAE for each setting.
#
# Do both settings reach the same error?

# %%
# solution
from sklearn.model_selection import cross_validate

cv_results = []
for kv_cache in [False, True]:
    results = cross_validate(
        TabICLRegressor(n_estimators=1, kv_cache=kv_cache, random_state=0),
        data,
        target,
        cv=ts_cv,
        scoring="neg_mean_absolute_error",
    )
    cv_results.append(
        pd.DataFrame(
            {
                "kv_cache": kv_cache,
                "fold": range(ts_cv.n_splits),
                "mae": -results["test_score"],
                "fit_time": results["fit_time"],
                "score_time": results["score_time"],
                "total_time": results["fit_time"] + results["score_time"],
            }
        )
    )
cv_results = pd.concat(cv_results, ignore_index=True)

cv_results.groupby("kv_cache")["mae"].agg(["mean", "std"])

# %% [markdown] tags=["solution"]
# Both settings reach a mean MAE of 36.9 bikes per hour with the same spread
# across folds, and the two means agree to the sixth decimal. The cache leaves
# the predictions unchanged.

# %% [markdown]
# ## Where does the compute time go?
#
# `cv_results` also holds the duration of `fit` and `predict` on each fold.
# Folds differ in how long they take, so we compare the two settings through
# the mean time across folds, with error bars for the standard deviation, as
# in the previous notebooks.
#
# Plot the mean fit time, predict time, and total time of each setting side by
# side, with error bars for the standard deviation across folds.
#
# Does the cache save compute time in cross-validation?

# %%
# solution
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

time_columns = {
    "fit_time": "fit",
    "score_time": "predict",
    "total_time": "fit + predict",
}
summary = cv_results.groupby("kv_cache")[list(time_columns)].agg(["mean", "std"])

for offset, (kv_cache, row) in zip([-0.1, 0.1], summary.iterrows()):
    plt.errorbar(
        [position + offset for position in range(len(time_columns))],
        row.xs("mean", level=1),
        yerr=row.xs("std", level=1),
        fmt="o",
        label=f"kv_cache={kv_cache}",
    )
plt.xticks(range(len(time_columns)), labels=time_columns.values())
plt.ylim(bottom=0)
plt.ylabel("Time per fold (s)")
plt.title("TabICL compute time, 800 training rows, 200 test rows")
plt.legend(loc="upper left")
plt.show()

# %% [markdown] tags=["solution"]
# The cache moves time from `predict` to `fit`. Without the cache, `fit` takes
# about a quarter of a second and `predict` about 5 seconds. That quarter of a
# second goes into reading the weights and preparing the ensemble members of
# the previous exercise, for example their column orders. With the cache,
# `fit` takes about 4.6 seconds, because it now runs the transformer over the
# 800 training rows of all 8 ensemble members. `predict` drops to about 1
# second, since only the 200 test rows remain to be processed.
#
# The two totals, about 5.5 seconds, overlap within their error bars.
# Cross-validation fits each model once and predicts once, so the stored work
# is never reused and the cache saves no time.

# %% [markdown]
# ## When the cache saves time
#
# In cross-validation, each fitted model predicts once. A deployed model
# instead keeps the same training rows and serves many predictions, so the
# work stored by the cache gets reused.
#
# Using the mean `fit_time` and `score_time` of each setting in `cv_results`,
# compute the total compute time of a model that is fitted once and then
# predicts 200 hours `n_calls` times, for `n_calls` from 1 to 10. Plot both
# settings on the same axes.
#
# From how many calls on does the cache save time?

# %%
# solution
import numpy as np

mean_times = cv_results.groupby("kv_cache")[["fit_time", "score_time"]].mean()
n_calls = np.arange(1, 11)

for kv_cache, times in mean_times.iterrows():
    plt.plot(
        n_calls,
        times["fit_time"] + n_calls * times["score_time"],
        "o-",
        label=f"kv_cache={kv_cache}",
    )
plt.xlabel("Number of predict calls")
plt.ylabel("Fit + predict time (s)")
plt.legend()
plt.show()

# %% [markdown] tags=["solution"]
# With a single call, both settings cost about the same, as in
# cross-validation. From the second call on, the cache saves time, because
# each extra call costs about 1 second instead of about 5. After 10 calls, the
# cached model has spent about 14 seconds, against 52 without the cache.
#
# The cache therefore pays off when the same training rows serve many
# predictions, as in a deployed model that keeps answering requests with the
# same context. It can also help when predict time matters more than fit
# time, although a GPU speeds up prediction too, with or without the cache. In
# return, the cache costs memory, since it stores the intermediate results of
# every layer for every ensemble member. In cross-validation, where each
# fitted model predicts once, we leave it off.
