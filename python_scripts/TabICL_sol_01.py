# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # How many ensemble members does TabICL need?
#
# In the previous notebook, TabICL took several seconds to predict 200 rows.
# Part of that time comes from a setting we never touched. TabICL doesn't run
# its transformer once per prediction. By default, it runs it eight times and
# averages the eight answers.
#
# Each of these runs, called an ensemble member, sees the same table in a
# different form. Members differ in the order of the columns, and in whether
# the numerical features go through a power transform. A single member's
# prediction shifts a little with these choices, and averaging several members
# smooths out the quirks of any one of them. The `n_estimators` parameter sets
# how many members TabICL averages. Each extra member costs one more pass of
# the transformer over the training rows and the rows to predict.
#
# More members should lower the error, but by how much, and at what price? In
# this exercise, we:
#
# * measure how TabICL's error and predict time change with `n_estimators`;
# * compare ensemble sizes fold by fold, to tell a small gain apart from noise;
# * find the point where extra members stop paying for their compute.

# %% [markdown]
# We reuse the bike sharing data and the time-aware splitter from the previous
# notebook, with two changes. In our runs, extra members made no measurable
# difference with 800-row training windows. They start to matter with shorter
# contexts, so we train on 200 rows. The gain also turns out to be small, so
# we need many folds to tell it apart from noise. The 40 test windows of 200
# hours cover the last 8,000 hours of the data, about eleven months.

# %%
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")
target = bike_sharing["count"]
data = bike_sharing.drop(columns="count")

ts_cv = TimeSeriesSplit(n_splits=40, gap=48, max_train_size=200, test_size=200)

# %% [markdown]
# ## Error and compute cost of each ensemble size
#
# `cross_validate` times every call to `fit` and `predict`. The first calls in
# a fresh Python session also pay one-off costs, for example reading the
# weights from disk for the first time and setting up PyTorch. These costs
# would inflate the timings of whichever ensemble size runs first. The next
# cell runs TabICL once on a small subset to absorb them, and defines the
# ensemble sizes to compare.
#
# ```{tip}
# The default grid takes about 5 minutes on a laptop CPU. With more compute,
# or a GPU, uncomment the full grid, which adds 32 members and about 3 more
# minutes.
# ```

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor

_ = (
    TabICLRegressor(random_state=0)
    .fit(data.iloc[:100], target.iloc[:100])
    .predict(data.iloc[100:120])
)

all_n_estimators = [1, 2, 4, 8, 16]
# all_n_estimators = [1, 2, 4, 8, 16, 32]

# %% [markdown]
# For each value in `all_n_estimators`, cross-validate a `TabICLRegressor`
# with `random_state=0`, using `ts_cv` and the mean absolute error as scoring.
# Collect the results in a dataframe named `cv_results`, with one row per
# ensemble size and fold. Its columns should be `n_estimators`, `fold`, `mae`,
# `fit_time`, and `score_time`. Then display the mean of each column for each
# ensemble size.
#
# Which of the three columns changes the most with the number of members?

# %%
# solution
from sklearn.model_selection import cross_validate

cv_results = []
for n_estimators in all_n_estimators:
    results = cross_validate(
        TabICLRegressor(n_estimators=n_estimators, random_state=0),
        data,
        target,
        cv=ts_cv,
        scoring="neg_mean_absolute_error",
    )
    cv_results.append(
        pd.DataFrame(
            {
                "n_estimators": n_estimators,
                "fold": range(ts_cv.n_splits),
                "mae": -results["test_score"],
                "fit_time": results["fit_time"],
                "score_time": results["score_time"],
            }
        )
    )
cv_results = pd.concat(cv_results, ignore_index=True)

cv_results.groupby("n_estimators")[["mae", "fit_time", "score_time"]].mean()

# %% [markdown] tags=["solution"]
# Fit time stays around a quarter of a second, whatever the number of members.
# `fit` never runs the transformer, so all the extra cost lands in `predict`.
# On our laptop, predicting takes 0.2 seconds with one member, 1.7 seconds with
# eight, and 3.5 seconds with sixteen. Predict time grows almost in proportion
# to the number of members. The mean MAE barely moves in comparison, from 52.9
# bikes per hour with one member to 51.7 with eight and 51.8 with sixteen.

# %% [markdown]
# ## Error and predict time as `n_estimators` grows
#
# Plot the mean MAE and the mean predict time against `n_estimators`, in two
# panels that share the x-axis. Add error bars for the standard deviation
# across folds. Since the ensemble sizes double from one value to the next, a
# base-2 logarithmic scale suits the x-axis.
#
# Can you tell from the error bars whether more members lower the MAE?

# %%
# solution
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

summary = cv_results.groupby("n_estimators")[["mae", "score_time"]].agg(
    ["mean", "std"]
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
for ax, column, ylabel in zip(
    axes, ["mae", "score_time"], ["MAE (bikes per hour)", "Predict time (s)"]
):
    ax.errorbar(
        summary.index,
        summary[column, "mean"],
        yerr=summary[column, "std"],
        fmt="o-",
    )
    ax.set_ylabel(ylabel)
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(all_n_estimators, labels=all_n_estimators)
axes[1].set_xlabel("n_estimators")
plt.show()

# %% [markdown] tags=["solution"]
# The MAE curve looks flat. Its error bars span about 17 bikes per hour on
# each side, because some test windows fall in quiet weeks and others in busy
# ones. With a single member, the MAE of a fold ranges from 21 to 104 bikes per
# hour. A gain of about 1 bike per hour is invisible at that scale.
#
# The predict time curve is easier to read. It keeps climbing, and it roughly
# doubles each time the number of members doubles.
#
# If you run the full grid, both curves stop moving between 16 and 32 members.
# TabICL builds one column order per column, and uses each order with and
# without the power transform. So it can build at most twice as many distinct
# members as there are columns. With our 12 columns, that ceiling is 24, and 32
# members already exceed it. The ceiling is even lower here, because TabICL
# first drops the columns that are constant in the training rows. Over 200
# hours, `year` never changes, and `season`, `holiday`, or `month` often don't
# either. Depending on the fold, 8 to 11 columns remain, and so 16 to 22
# distinct members. TabICL ignores any member beyond that. On the 17 folds
# that keep only 8 columns, 32 members return exactly the same predictions as
# 16, in about the same time.

# %% [markdown]
# ## Comparing ensemble sizes fold by fold
#
# The error bars in the previous plot mostly reflect how hard each test window
# is, and not the effect of `n_estimators`. Every ensemble size sees the same
# folds, so we can compare them within each fold and remove that spread.
#
# For each fold, compute the gain of each ensemble size over a single member,
# that is, the MAE with one member minus the MAE with `n_estimators` members.
# A positive gain means a lower error. Plot the mean gain against
# `n_estimators`, with error bars for the standard deviation across folds, as
# in the previous plot.
#
# Does the gain stand out from its spread across folds?

# %%
# solution
mae_per_fold = cv_results.pivot(index="fold", columns="n_estimators", values="mae")
gain = -mae_per_fold.sub(mae_per_fold[1], axis=0)

plt.errorbar(gain.columns, gain.mean(), yerr=gain.std(), fmt="o-")
plt.axhline(0, color="black", linestyle="--")
plt.xscale("log", base=2)
plt.xticks(all_n_estimators, labels=all_n_estimators)
plt.xlabel("n_estimators")
plt.ylabel("MAE gain over one member\n(bikes per hour)")
plt.show()

# %% [markdown] tags=["solution"]
# Eight members lower the MAE by 1.2 bikes per hour on average, and sixteen
# members by 1.0. The error bars span about 4 bikes per hour on each side,
# four times less than in the previous plot, because comparing within each
# fold removes how hard each window is. They still cross zero, so the gain
# varies from fold to fold, and it is small, about 2% of the error.
#
# Going from 8 to 16 members brings nothing, and the mean gain even drops by
# 0.1 bikes per hour. The error has saturated by 8 members, but predict time
# keeps doubling. Two and four members bring about the same mean gain, 1.0
# and 0.9 bikes per hour, but with wider error bars, about 5 and 4.5 bikes per
# hour on each side against 4 for eight members.
#
# On this data, the default of 8 members buys a small gain, at about seven
# times the predict time of a single member. Whether that trade is worth it
# depends on how much predict time matters.

# %% [markdown]
# These curves suggest a rule of thumb. Increase `n_estimators` until you hit
# one of two limits, whichever comes first. The first limit is the plateau,
# where extra members stop lowering the error. You find it by measuring, as we
# did here, and it sits around 8 members on this data. The second limit is the
# ceiling of twice the number of non-constant columns. Beyond it, TabICL
# builds no new members. Prefer even values, because members come in pairs,
# one with and one without the power transform.
#
# The plateau depends on the data. With 800-row windows, as in the previous
# notebook, we found no measurable gain at all, so the plateau starts at a
# single member. Keep this in mind when later notebooks trim the ensemble to a
# single member to save time.
