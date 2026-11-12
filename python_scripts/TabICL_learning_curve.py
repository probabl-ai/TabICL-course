# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # When does a tabular foundation model pay off?
#
# In the previous notebook, TabICL predicted hourly bike rentals with a lower
# average error than `HistGradientBoostingRegressor` (HGBT) on 800-row training
# windows, even after a 16-candidate random search for HGBT. TabICL's edge comes
# from pretraining, which should matter most when our own data is scarce.
#
# In this notebook, we check how the gap between both models evolves as the
# training window grows, using a learning curve.

# %%
import pandas as pd

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")

target_name = "count"
target = bike_sharing[target_name]
data = bike_sharing.drop(columns=target_name)

# %% [markdown]
# ## Learning curve with time-based splits
#
# A **learning curve** repeats the same cross-validation for a growing number
# of training samples and tracks how the test error changes. Here we only
# compare the test MAE of both models, not their training errors.
#
# `LearningCurveDisplay` builds its smaller training sets by keeping the first
# rows of each training fold. In a time series, these are the oldest rows, the
# furthest from the test set. Instead, we rebuild the `TimeSeriesSplit` for
# each training size by changing `max_train_size`, so that each model always
# reads the most recent hours before the gap. Since `test_size` and `n_splits`
# stay fixed, both models are evaluated on the same test hours at every
# training size. These are the last 5 of the previous notebook's 10 test
# windows, so errors at 800 rows differ from the ones we saw there.
#
# TabICL's memory and prediction time grow with the size of its context, so we
# stop it at 3,200 rows to stay within a laptop's reach. HGBT is cheap enough
# to go up to 12,800 rows, about a year and a half of rentals. HGBT keeps its
# default hyperparameters, because the smallest windows are too short to nest
# a time-aware search inside them, and tuning gained it little in the previous
# notebook.
#
# ```{tip}
# The TabICL part of this cell takes about 6 minutes to run on a CPU. On a slow
# machine, uncomment the line that stops TabICL at 800 rows, which brings it
# down to under 2 minutes. If you have a CUDA GPU, which TabICL uses
# automatically, you can instead uncomment the line that runs TabICL up to
# 12,800 rows, like HGBT. The discussion below refers to the curve up to 3,200
# rows.
# ```

# %%
# %pip install -q tabicl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from tabicl import TabICLRegressor

models = {
    "HGBT": HistGradientBoostingRegressor(random_state=0),
    "TabICL": TabICLRegressor(random_state=0),
}
models["TabICL"].fit(data.iloc[:100], target.iloc[:100])  # warm-up

train_sizes = {
    "HGBT": [100, 200, 400, 800, 1_600, 3_200, 6_400, 12_800],
    "TabICL": [100, 200, 400, 800, 1_600, 3_200],
}
# train_sizes["TabICL"] = [100, 200, 400, 800]
# train_sizes["TabICL"] = train_sizes["HGBT"]

learning_curve_mae = {}
learning_curve_time = {}
for name, model in models.items():
    learning_curve_mae[name] = []
    learning_curve_time[name] = []
    for train_size in train_sizes[name]:
        cv = TimeSeriesSplit(
            n_splits=5, gap=48, max_train_size=train_size, test_size=200
        )
        results = cross_validate(
            model, data, target, cv=cv, scoring="neg_mean_absolute_error"
        )
        learning_curve_mae[name].append(-results["test_score"])
        learning_curve_time[name].append(results["fit_time"] + results["score_time"])

# %% [markdown]
# Notice that the cell above calls TabICL's `fit` once before the loop. As we
# saw in the previous notebook, the first call downloads the pretrained
# weights. Doing it before the loop keeps the download out of the timings, so
# that no fold pays a cost the others don't, and differences in time come only
# from the model's computations.

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 200
model_colors = {"HGBT": "tab:blue", "TabICL": "tab:orange"}

_, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for name, mae in learning_curve_mae.items():
    mae = np.array(mae)
    ax.errorbar(
        train_sizes[name],
        mae.mean(axis=1),
        yerr=mae.std(axis=1),
        fmt="o-",
        capsize=3,
        color=model_colors[name],
        label=name,
        alpha=0.7,
    )
ax.set(
    xscale="log",
    xlabel="Number of training rows",
    ylabel="Test mean absolute error",
    title="Learning curve for HGBT and TabICL",
)
ax.legend()
plt.show()

# %% [markdown]
# TabICL keeps a lower average error at every training size where both run,
# although the error bars overlap at some sizes. With a few hundred rows, the
# gap is large, and HGBT needs 3,200 rows to come close to the error TabICL
# reaches with 400. As the window grows, HGBT's error keeps falling up to
# 3,200 rows and the gap narrows, while TabICL's error barely improves beyond
# 400 rows.
#
# Why does TabICL win when data is scarce? With a few hundred rows, many
# relationships fit the training data, so the inductive bias of a model plays
# a larger role. The inductive bias is the kind of relationship a model type
# favors by construction. Gradient boosting builds step functions from splits
# on single features, and estimates each step from the few rows that fall in
# it. TabICL's prior mixes many kinds of relationships, such as smooth curves,
# linear trends, plateaus and tree-like splits, and TabICL infers from the
# context which kind fits the table. We will look at this prior in a dedicated
# video. Where the truth is smooth, TabICL can interpolate between a few points
# where trees would draw noisy steps. As the window grows, the data pins down
# the relationship on its own, the inductive bias matters less, and HGBT
# closes part of the gap.
#
# Data can also be scarce in only part of a table. With imbalanced classes,
# the rare class has few rows even in a large table, so the inductive bias
# matters most there. We will meet this case in a later notebook on card
# fraud.
#
# Each additional row of context also costs TabICL more memory and more
# prediction time. `cross_validate` records the time spent fitting and scoring
# each fold, and we plot their sum, the time it takes to train a model and
# predict the 200 test hours.

# %%
_, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for name, fit_predict_time in learning_curve_time.items():
    fit_predict_time = np.array(fit_predict_time)
    ax.errorbar(
        train_sizes[name],
        fit_predict_time.mean(axis=1),
        yerr=fit_predict_time.std(axis=1),
        fmt="o-",
        capsize=3,
        color=model_colors[name],
        label=name,
        alpha=0.7,
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Number of training rows",
    ylabel="Fit + predict time per fold (s)",
    title="Wall-clock time for HGBT and TabICL",
)
ax.legend()
plt.show()

# %% [markdown]
# HGBT needs about a second per fold at most, and its time grows much more
# slowly than the number of rows. TabICL takes a few seconds per fold up to 800
# rows, then its time roughly doubles each time the training window doubles,
# reaching about 27 seconds per fold at 3,200 rows. TabICL learns in context, so
# it processes the whole training window again every time it predicts.
#
# The exact timings depend on your hardware, but the trend should hold. At
# 3,200 rows, TabICL is more than an order of magnitude slower than HGBT
# trained on 12,800 rows.
#
# A TFM pays off most when data is scarce. As the training window grows,
# TabICL's advantage shrinks, while its compute cost grows much faster than
# HGBT's.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * built a learning curve for time series by shrinking `max_train_size` in
#   `TimeSeriesSplit`, so every training window ends right before the gap;
# * compared the test MAE of HGBT and TabICL from 100 to 12,800 training rows;
# * compared their fit + predict wall-clock time over the same training sizes;
# * saw TabICL win clearly when data is scarce, and its advantage shrink as the
#   training window grows, while its compute cost grows much faster than
#   HGBT's;
# * explained the advantage on small data by the inductive bias, which matters
#   most when few rows constrain the relationship.
