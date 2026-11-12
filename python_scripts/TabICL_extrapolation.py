# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Extrapolating with TabICL
#
# So far we have only asked TabICL to predict rows that fall within the range
# of its training context. In practice, a new row can fall outside that range,
# and the model still returns a prediction for it.
#
# In this notebook we use a small synthetic dataset with two features, so that
# we can plot the full prediction surface, and we:
#
# * look at the kinds of functions TabICL's prior generates during
#   pretraining, and which parts of our data they cover;
# * fit TabICL on that data and ask it to predict a region with no training
#   rows;
# * compare it to a `HistGradientBoostingRegressor` in that region, and
#   recall why tree-based models cannot extrapolate without feature
#   engineering.

# %% [markdown]
# ## The landscape dataset
#
# `landscape_dataset.csv` is a synthetic dataset with two features, `x1` and
# `x2`, both between -6 and 6, and a `target` we can read as the elevation at
# each point.

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
# The target combines three patterns. The whole surface tilts upward as `x1`
# and `x2` grow, which is a linear trend. Near the center, there is a sharp
# circular drop. On top of both, a sine of `x1` multiplied by a cosine of `x2`
# makes the surface oscillate across the whole square.

# %% [markdown]
# ## TabICL's prior
#
# TabICL's weights were pretrained on millions of synthetic tables drawn from
# a *prior*, a random generator of tables. Each table comes from a random
# causal graph where every variable is computed from its parents by one of
# eight kinds of random function: small neural networks, tree ensembles,
# linear and quadratic functions, smooth random surfaces, functions with flat
# plateaus, and products of these. During pretraining, the model was given
# some rows of such a table as context and had to predict the target of other
# rows.
#
# The prior covers two of the three patterns in our data. Linear functions
# produce linear trends, and plateau-like and tree-based functions produce
# round regions with sharp edges, like our circular drop.
#
# None of the eight kinds of random function is periodic, though. Sine
# appears only as one of a few dozen activations that a neural network may
# draw, and no pretraining table looked like our dataset. As we will see, the
# model still handles the oscillation. It pieces it together from the kinds
# of structure it does know, guided by the examples we hand it in-context at
# prediction time.
#
# ```{note}
# For further reading, the TabICLv2 prior is described in Section 5 and
# Figure 4 of the [TabICLv2 paper](https://arxiv.org/abs/2602.11139), and the
# full lists of random functions and activations are in Appendices E.8 and
# E.9. The [original TabICL paper](https://arxiv.org/abs/2502.05564)
# describes the earlier prior that TabICLv2 builds on.
# ```

# %% [markdown]
# ## Predicting outside the training range
#
# The training data has no row with `x1` above 6. We build a grid of points
# where `x1` goes up to 12, so that the strip between 6 and 12 has no
# training rows at all.

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
# As before, `fit` only stores the training rows as context, and TabICL reads
# them when we call `predict`. That call is the same whether or not the rows
# we predict on fall outside the training range, and the model has no way to
# tell whether they do.
#
# We use a single ensemble member, `n_estimators=1`, to keep each call to
# `predict` fast.

# %%
# %pip install -q tabicl
from tabicl import TabICLRegressor

tabicl = TabICLRegressor(n_estimators=1, random_state=0)
tabicl.fit(data, target)
grid_prediction_tabicl = tabicl.predict(grid_data).reshape(grid_x1.shape)

plot_landscape_surface(
    grid_prediction_tabicl, "TabICL (viridis: observed, red: x1 > 6)"
)

# %% [markdown]
# Inside the training square, TabICL recovers all three patterns, including
# the oscillation, even though it never saw this surface during pretraining.
#
# Past `x1 = 6`, the surface bends smoothly away from the observed region and
# still rises and falls with `x2`, but along `x1` it mostly levels off. It
# continues neither the trend nor the oscillation, because there are no
# examples in that strip to read them from.

# %% [markdown]
# For comparison, we fit a `HistGradientBoostingRegressor` on the same rows
# and predict on the same grid.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

hgbt = HistGradientBoostingRegressor(random_state=0)
hgbt.fit(data, target)
grid_prediction_hgbt = hgbt.predict(grid_data).reshape(grid_x1.shape)

plot_landscape_surface(grid_prediction_hgbt, "HGBT (viridis: observed, red: x1 > 6)")

# %% [markdown]
# Past `x1 = 6`, HGBT's surface is completely flat along `x1`. This follows
# from how regression trees work, so we would see it with any fit. Every split
# threshold a tree learns lies inside the training range, so a point beyond
# the largest `x1` in the training set falls on the same side of every split
# on `x1`, and its prediction no longer depends on `x1`. Even the linear
# trend, the simplest of the three patterns, stops at the edge of the data.
#
# We already saw this limitation earlier in the course. Without proper
# feature engineering, tree-based models extrapolate poorly. The usual fixes
# add what we know about the problem to the features or to the model before
# fitting:
#
# * if we expect a trend to continue, a linear model, alone or combined with
#   the trees, can carry it beyond the training range;
# * if we expect a periodic pattern, periodic features such as a sine and
#   cosine of `x1`, or a `SplineTransformer(extrapolation="periodic")`, let
#   the trees reuse what they learned on one period for the next one.
#
# Both fixes only work if we already know which structure to expect and pick
# the right encoding for it.

# %% [markdown]
# TabICL's continuation is smoother than the tree's plateau, and it needed no
# feature engineering. That does not make it correct. Past `x1 = 6`, no
# in-context example constrains the prediction, and TabICL returns its guess
# of how surfaces like this one usually continue. Here, that guess drops both
# the trend and the oscillation, as the tree did. Pretraining helps TabICL
# read structure from the examples we give it, but it cannot recover
# structure in a region where we gave no examples.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * looked at a synthetic dataset combining a linear trend, a sharp circular
#   drop and an oscillation built from sines and cosines;
# * saw that TabICL's prior generates tables from eight kinds of random
#   function, which cover the trend and the drop but none of which is
#   periodic, and that the model pieces the oscillation together from the
#   examples we give it in-context;
# * asked TabICL and a `HistGradientBoostingRegressor` to predict a strip of
#   `x1` with no training rows;
# * saw HGBT's predictions go flat along `x1` past the training range, as
#   follows from how tree splits work, and recalled that trees need feature
#   engineering to extrapolate;
# * saw TabICL's predictions bend smoothly past the training range but also
#   mostly level off along `x1`, since there are no in-context examples there
#   to read from.
