# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # A tabular foundation model doesn't know where the map ends
#
# So far we've only asked TabICL to predict inside the world it was shown,
# rows drawn from roughly the same range as its training context. Real
# tables are rarely that polite. A new row can land past the edge of
# anything the model has read, and another row can arrive with one of its
# values simply missing.
#
# Both situations ask the model to say something about a coordinate it
# never directly observed. In this notebook we look at both, on a small
# synthetic surface simple enough to plot in full, and end up finding they
# are really the same question asked twice.
#
# In this notebook, we:
#
# * fit TabICL on a synthetic landscape and ask it to predict a region it
#   never trained on, and compare its behaviour there to a
#   `HistGradientBoostingRegressor`;
# * introduce heavy missing values into that same landscape, and watch
#   TabICL and HGBT predict straight through the gaps, no imputation step
#   required;
# * make that native handling explicit with TabICL's own imputer, and see
#   that predicting a new row and filling in a missing value are the same
#   operation, from the model's point of view.

# %% [markdown]
# ## What happens once we ask for a point off the map?
#
# `landscape_dataset.csv` is a synthetic dataset: two coordinates, `x1` and
# `x2`, and a `target` we can read as the elevation at that point. Every row
# in it has both coordinates between -6 and 6.

# %%
import pandas as pd

landscape = pd.read_csv("../datasets/landscape_dataset.csv")
landscape.describe()

# %%
target_name = "target"
target = landscape[target_name]
data = landscape.drop(columns=target_name)

# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(projection="3d")
ax.scatter(data["x1"], data["x2"], target, c=target, cmap="viridis", s=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("target")
plt.show()

# %% [markdown]
# Rolling hills cover most of the square, with a sudden, cliff-like drop
# near the center. Nowhere does `x1` go past 6. We are about to ask both
# models for exactly that, `x1` between 6 and 9, a strip of the surface
# neither one has ever been trained on.

# %%
import numpy as np

extrapolation_boundary = data["x1"].max()
grid_1d_x1 = np.linspace(-6, 12, 60)
grid_1d_x2 = np.linspace(-6, 6, 60)
grid_x1, grid_x2 = np.meshgrid(grid_1d_x1, grid_1d_x2)
grid_data = pd.DataFrame({"x1": grid_x1.ravel(), "x2": grid_x2.ravel()})


def plot_landscape_surface(grid_prediction, title):
    observed = np.where(grid_x1 <= extrapolation_boundary, grid_prediction, np.nan)
    extrapolated = np.where(grid_x1 > extrapolation_boundary, grid_prediction, np.nan)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(grid_x1, grid_x2, observed, cmap="viridis", alpha=0.9, linewidth=0)
    ax.plot_surface(grid_x1, grid_x2, extrapolated, cmap="Reds", alpha=0.9, linewidth=0)
    ax.scatter(data["x1"], data["x2"], target, color="black", s=4, alpha=0.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("target")
    ax.set_title(title)
    plt.show()


# %% [markdown]
# As before, `fit` only stores the training rows as context, and `predict` is
# where TabICL actually reads them and answers. Nothing about that call changes
# when some of the questions we ask fall outside the training range, the model
# has no way of even knowing that they do.

# %%
# %pip install -q tabicl
from sklearn.metrics import mean_absolute_error
from tabicl import TabICLRegressor

tabicl = TabICLRegressor(random_state=0)
tabicl.fit(data, target)
grid_prediction_tabicl = tabicl.predict(grid_data).reshape(grid_x1.shape)
mae_tabicl_clean = mean_absolute_error(
    target, tabicl.predict(data)
)  # kept for later, before we refit on missing data

plot_landscape_surface(
    grid_prediction_tabicl, "TabICL (viridis: observed, red: x1 > 6)"
)

# %% [markdown]
# For comparison, we fit a `HistGradientBoostingRegressor` on the same
# rows and ask it the same question.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

hgbt = HistGradientBoostingRegressor(random_state=0)
hgbt.fit(data, target)
grid_prediction_hgbt = hgbt.predict(grid_data).reshape(grid_x1.shape)
mae_hgbt_clean = mean_absolute_error(
    target, hgbt.predict(data)
)  # kept for later, before we refit on missing data

plot_landscape_surface(grid_prediction_hgbt, "HGBT (viridis: observed, red: x1 > 6)")

# %% [markdown]
# HGBT's surface goes completely flat past `x1 = 6`. That's not a quirk of
# this particular fit, it follows directly from how regression trees work.
# Every split threshold a tree learns sits somewhere inside the training
# range, so any point past the largest `x1` it has seen lands in the same
# rightmost leaf as every other point out there. Its prediction cannot
# change with `x1` anymore, no matter how far past 6 we push it.
#
# TabICL's surface doesn't freeze the same way. It keeps some texture as
# `x1` grows, still rising and falling with `x2` the way the observed part
# does. Pretraining on a huge variety of synthetic functions gives it some
# notion that a surface usually keeps doing something instead of going
# flat, so it produces a continuation that at least looks plausible, rather
# than a plateau.
#
# Plausible is not the same as correct, though. Extrapolation asks the
# model about a coordinate it has never been shown a single example of, and
# no amount of pretraining changes that. TabICL's guess merely looks less
# broken than a tree's exact plateau, it isn't a confirmed right answer.

# %% [markdown]
# ## Native support for missing values
#
# Extrapolation pushes a coordinate outside the entire training square. A
# missing value is a gentler version of the same idea. One coordinate of a
# row is unknown, but the row itself still sits in territory the model
# already knows well, surrounded by complete rows on every side.

# %%
# Randomly mask one feature per row for most of the rows.
rng = np.random.default_rng(0)

# For each selected row, mask exactly one of the two features.
mask = np.zeros(data.shape, dtype=bool)
masked_rows = rng.choice(len(data), size=600, replace=False)
masked_col = rng.integers(0, 2, size=len(masked_rows))
mask[masked_rows, masked_col] = True

data_missing = data.copy()
data_missing[mask] = np.nan

is_observed = ~mask.any(axis=1)
is_partial = mask.any(axis=1)

print(
    f"Rows: {len(data)} total, {is_partial.sum()} with missing values, "
    f"{is_observed.sum()} fully observed"
)
print(f"Cells: {mask.sum()} / {mask.size} missing ({100 * mask.mean():.0f} %)")

# %%
tabicl.fit(data_missing, target)
grid_prediction_tabicl = tabicl.predict(grid_data).reshape(grid_x1.shape)

plot_landscape_surface(
    grid_prediction_tabicl, "TabICL (viridis: observed, red: x1 > 6)"
)

# %%
hgbt.fit(data_missing, target)
grid_prediction_hgbt = hgbt.predict(grid_data).reshape(grid_x1.shape)

plot_landscape_surface(grid_prediction_hgbt, "HGBT (viridis: observed, red: x1 > 6)")

# %% [markdown]
# Both fits ran without any extra step. Neither TabICL nor
# `HistGradientBoostingRegressor` needed the gaps filled in first, both
# accept `NaN` directly in their training data. Not every scikit-learn
# estimator can do that, a `Ridge` regression would raise an error on the
# very same call. By construction, every incomplete row still keeps one of
# its two coordinates, so there is always something left to reason from.

# %%
mae_tabicl_missing = mean_absolute_error(target, tabicl.predict(data))
mae_hgbt_missing = mean_absolute_error(target, hgbt.predict(data))

print(
    f"TabICL MAE: {mae_tabicl_clean:.2f} clean -> {mae_tabicl_missing:.2f} with gaps in training"
)
print(
    f"HGBT MAE:   {mae_hgbt_clean:.2f} clean -> {mae_hgbt_missing:.2f} with gaps in training"
)

# %% [markdown]
# Both models get worse once most of their training rows arrive with a
# coordinate missing, unsurprisingly, they now have less to go on. HGBT
# degrades more, though, its error roughly triples, while TabICL's a bit
# more than doubles. TabICL's pretraining again gives it a slight edge, one
# missing coordinate still leaves enough of a pattern in the rest of the
# row, and in the surrounding rows, to reason from.
#
# Neither model asked us to fill anything in beforehand, and that is only
# possible because, internally, coping with a missing `x1` already requires
# guessing a plausible value for it from whatever the row does have. That
# guess is an imputation, whether we call it that or not. Native support
# for missing values isn't a separate feature bolted onto these models,
# it's the same read-the-landscape reasoning we're about to make explicit.

# %% [markdown]
# So far `x1` and `x2` were always "the features" and `target` was always
# "the answer". Let's make that implicit reasoning explicit. `tabicl` also
# ships `TabICLUnsupervised`, an estimator that drops that distinction
# entirely. It fits on a plain table, no target column singled out, and its
# `impute` method fills whatever is missing in a row by conditioning on
# whatever else that row happens to have, `x1`, `x2`, or `target`, whichever
# is present. `target` is never masked, so it stays available as one more
# known coordinate for every incomplete row, exactly as it was when HGBT and
# TabICL predicted through the gaps a moment ago.

# %%
from tabicl import TabICLUnsupervised

table_missing = data_missing.assign(target=target)

imputer = TabICLUnsupervised(random_state=0)
imputer.fit(table_missing)
data_imputed = imputer.impute(table_missing)

# %% [markdown]
# `fit` stores all 800 rows, missing cells included, as context, nothing in
# that call says which column is the target. `impute` then revisits every
# row that has a gap and fills it with the value that best matches
# everything else it has seen, low `temperature` by default, so it reports
# its single best guess rather than a random draw among plausible values.
#
# `landscape_dataset.csv` holds these same 800 rows with nothing missing,
# so we can check those guesses against the truth.

# %%
missing_x1_mask = table_missing["x1"].isna()
missing_x2_mask = table_missing["x2"].isna()

true_x1 = landscape.loc[missing_x1_mask, "x1"]
true_x2 = landscape.loc[missing_x2_mask, "x2"]
predicted_x1 = data_imputed[missing_x1_mask.to_numpy(), 0]
predicted_x2 = data_imputed[missing_x2_mask.to_numpy(), 1]

mae_x1 = mean_absolute_error(true_x1, predicted_x1)
mae_x2 = mean_absolute_error(true_x2, predicted_x2)
print(f"x1 imputation MAE: {mae_x1:.2f}")
print(f"x2 imputation MAE: {mae_x2:.2f}")

# %% [markdown]
# Is that good? As a point of reference, we compare against the simplest
# possible guess, filling every missing value with the average of that
# column.

# %%
naive_x1 = np.full(len(true_x1), table_missing["x1"].mean())
naive_x2 = np.full(len(true_x2), table_missing["x2"].mean())

print(f"x1 mean-fill MAE: {mean_absolute_error(true_x1, naive_x1):.2f}")
print(f"x2 mean-fill MAE: {mean_absolute_error(true_x2, naive_x2):.2f}")

# %% [markdown]
# TabICL's guesses land two to three times closer to the truth than the
# column average. It isn't recalling a typical `x1`, it reads the landscape
# at the point described by that row's other coordinate and its `target`, and
# reports what it finds there, the same reasoning it used to predict
# `target` from `x1` and `x2` at the start of this notebook, with the roles
# of "known" and "unknown" simply reshuffled.
#
# Drop `target` from that table and hand the imputer only `x2` to recover
# `x1` from, and it stops beating the column average altogether. `x1` and
# `x2` were drawn independently of each other, only `target` ties them
# together, so without it there is nothing left to read off. Imputation is
# only as good as the signal actually available in the columns we hand it,
# exactly like any other prediction task.
#
# `impute` isn't doing anything exotic, either. TabICL's own documentation
# describes it as a MICE-like procedure, the same idea behind scikit-learn's
# `IterativeImputer`. Each column that has missing values gets modeled as a
# function of the others, and those guesses get refined over a couple of
# passes.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * fit TabICL and a `HistGradientBoostingRegressor` on a synthetic
#   landscape, then asked both to predict a strip of `x1` neither had
#   trained on;
# * saw HGBT's predictions go exactly flat past the training range, a
#   direct consequence of how tree splits work, while TabICL's kept some
#   texture without being provably correct;
# * masked most rows down to a single known coordinate and watched TabICL
#   and HGBT predict `target` straight through the gaps with no imputation
#   step, HGBT's error degrading noticeably more than TabICL's;
# * made that native handling explicit with `TabICLUnsupervised.impute`,
#   which fits with no target column singled out, and checked its guesses
#   against the true values and a mean-fill baseline;
# * saw that imputation is only as good as the signal actually available,
#   dropping `target` from the imputer's table collapsed its guesses back
#   to the mean-fill baseline, since `x1` and `x2` alone carry nothing
#   about each other;
# * concluded that predicting a new row and filling in a missing value are
#   the same operation for a TFM, reading a landscape at a point it wasn't
#   directly given.
