# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
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
# Write your code here.

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
# Write your code here.

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
# Write your code here.
