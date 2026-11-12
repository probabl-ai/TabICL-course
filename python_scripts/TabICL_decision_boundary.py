# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Peeking inside predicted probabilities
#
# A classifier does more than pick a class. Its `predict_proba` method returns
# a probability for each class, and later notebooks rely on these
# probabilities to make decisions. Before that, we build some intuition on
# toy data with only two features, where we can draw the probability that a
# model predicts at every point of the plane.
#
# We compare three models, `LogisticRegression`, `HistGradientBoostingClassifier`
# (HGBT) and TabICL. By the end, you should be able to:
#
# - Read a probability map drawn with a diverging colormap.
# - Recognize the inductive bias of each model in its probability map.
# - See how each model behaves away from the training data and around a rare
#   class.
# - Compare the predicted probabilities of several models with the log loss.

# %% [markdown]
# ## Reading a probability map
#
# As in the [scikit-learn MOOC](https://skolar.probabl.ai/en/a/2369819257005011388),
# we draw `predict_proba` with
# `DecisionBoundaryDisplay` and a diverging colormap. Blue means a low
# probability of the positive class, red a high one, and a probability of 0.5
# is mapped to white. The darker the colour, the more sure the model is of
# its prediction, whichever the class. The dashed line marks the probability
# of 0.5, where `predict` switches from one class to the other.
#
# The maps cover a wider area than the data, so they also show how each model
# extrapolates where it has seen no rows.
#
# ## Inductive bias
#
# The learning curve notebook introduced the inductive bias, the kind of
# relationship a model type favors by construction. With few rows, many
# relationships fit the data, and the inductive bias decides what the model
# predicts between and beyond them. In two dimensions, we can see it.
#
# - Logistic regression applies a sigmoid to a linear function of the
#   features. Its probability only changes along one direction, and its
#   decision boundary is a straight line.
# - Gradient boosting adds up many trees. Each tree splits on one feature at
#   a time, so the probability is a step function made of boxes aligned with
#   the axes.
# - TabICL's prior mixes many kinds of relationships, such as smooth curves,
#   linear trends, plateaus and tree-like splits, and TabICL infers from the
#   context which kind fits the table.

# %%
# %pip install -q tabicl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tabicl import TabICLClassifier


def make_models():
    return {
        "Logistic regression": LogisticRegression(),
        "HGBT": HistGradientBoostingClassifier(random_state=0),
        # A single ensemble member is enough here, and keeps the maps fast
        "TabICL": TabICLClassifier(n_estimators=1, random_state=0),
    }


def plot_probabilities(X, y):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.5), sharey=True, layout="constrained"
    )
    for ax, (name, model) in zip(axes, make_models().items()):
        model.fit(X, y)
        display = DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            response_method="predict_proba",
            plot_method="pcolormesh",
            grid_resolution=80,
            eps=1,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            alpha=0.8,
            ax=ax,
        )
        surface = display.surface_
        display.plot(
            ax=ax, plot_method="contour", levels=[0.5], colors="black",
            linestyles="--",
        )
        ax.scatter(*X.T, c=y, cmap="RdBu_r", vmin=0, vmax=1, edgecolors="white")
        ax.set(title=name, xlabel="Feature #0", ylabel="")
    axes[0].set_ylabel("Feature #1")
    fig.colorbar(surface, ax=axes, label="P(positive)")
    plt.show()


# %% [markdown]
# To compare the probabilities with a number, we use the log loss, which
# heavily penalizes a confident wrong probability. A repeated stratified
# k-fold gives 25 train-test splits, the same ones for every model, so we can
# subtract their scores split by split.

# %%
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)


def compare_log_loss(X, y, true_proba=None):
    scores = pd.DataFrame(
        {
            name: -cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
            for name, model in make_models().items()
        }
    )
    if true_proba is not None:
        scores["true probability"] = [
            log_loss(y[test], true_proba[test]) for _, test in cv.split(X, y)
        ]
    for name in ["Logistic regression", "HGBT"]:
        scores[f"{name} minus TabICL"] = scores[name] - scores["TabICL"]
    return scores.agg(["mean", "std"]).T.round(3)


# %% [markdown]
# ## Two moons
#
# Two interleaved half-moons, one per class, with some noise so that they
# overlap.

# %%
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
plot_probabilities(X, y)

# %% [markdown]
# Each map shows the inductive bias of its model. Logistic regression draws a
# straight boundary across the moons, with a wide white strip where it is
# unsure. HGBT tiles the plane with boxes aligned with the axes, and its
# boundary is a staircase. TabICL draws a curved boundary that follows the
# gap between the moons.
#
# Away from the data, each model extends its own kind of relationship. The
# probability of logistic regression keeps growing darker along the same
# direction. HGBT extends its outermost boxes, including a pale red box in
# the top right corner, where there is no row at all. TabICL extends its
# curves.
#
# A dark colour means that the model is sure, not that it is right. The log
# loss checks this on the test rows of each split.

# %%
compare_log_loss(X, y)

# %% [markdown]
# TabICL has the lowest log loss on average. Logistic regression is clearly
# worse, since its difference with TabICL is about three times its standard
# deviation. The difference between HGBT and TabICL is smaller than its
# standard deviation, so it varies a lot from one split to another.

# %% [markdown]
# ## An imbalanced XOR
#
# Two large negative blobs sit on one diagonal, and two small positive blobs
# of 8 rows each on the other one. The blobs overlap, so some rows are
# ambiguous.

# %%
from sklearn.datasets import make_blobs

X, blob = make_blobs(
    n_samples=[50, 50, 8, 8],
    centers=[(-1, 1), (1, -1), (1, 1), (-1, -1)],
    cluster_std=0.5,
    random_state=0,
)
y = (blob >= 2).astype(int)
plot_probabilities(X, y)

# %% [markdown]
# No straight line separates the two diagonals, so logistic regression
# cannot represent this relationship. It predicts about the same low
# probability everywhere, close to the share of positive rows, 16 out of 116.
#
# HGBT isolates each positive blob in a red box, with sharp edges and pale
# strips that reach into empty areas. TabICL also finds the two positive
# blobs, but with curved boundaries. Both models are sure in the empty
# corners, far from any row.

# %%
compare_log_loss(X, y)

# %% [markdown]
# Logistic regression has by far the highest log loss, as expected from a
# model that cannot represent the relationship. TabICL has the lowest on
# average, but the difference with HGBT is about as large as its standard
# deviation. With only 16 positive rows, a test fold holds 3 or 4 of them,
# and a single confident mistake on one of them weighs a lot.

# %% [markdown]
# ## A rare class along a curve
#
# Here the positive rows lie along a thin sine-shaped band. Even on the
# centre line of the band, a row is positive only half of the time. The data
# is synthetic, so we know the true probability, and its log loss is the
# best any model could reach.


# %%
def sine_band_proba(X, amplitude=1.5, frequency=1.2, half_width=0.35, sharpness=12.0):
    distance_to_curve = np.abs(X[:, 1] - amplitude * np.sin(frequency * X[:, 0]))
    inside_band = 1 / (1 + np.exp(-sharpness * (half_width - distance_to_curve)))
    return 0.5 * inside_band


rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(400, 2))
proba = sine_band_proba(X)
y = rng.binomial(1, proba)
print(f"{y.sum()} positive rows out of {len(y)}")

plot_probabilities(X, y)

# %% [markdown]
# Only 25 of the 400 rows are positive, scattered along the band. Logistic
# regression again predicts about
# the same low probability everywhere.
#
# HGBT draws a few small dark red boxes around groups of positive rows, with
# probabilities close to 1, although the true probability never exceeds 0.5.
# Between these boxes, the rest of the band is as dark blue as the empty
# areas. TabICL draws a faint ridge that follows the band, with
# probabilities mostly below 0.5, and keeps lighter colours outside the
# data.

# %%
compare_log_loss(X, y, true_proba=proba)

# %% [markdown]
# TabICL's log loss comes closest to the one of the true probability on
# average, but none of the models reaches it. HGBT pays for its confident boxes, and logistic
# regression for missing the band entirely. Logistic regression has the
# smallest spread across splits, because it predicts about the same
# probability everywhere.
#
# When positive rows are rare, the inductive bias decides what the model
# predicts between them. HGBT's boxes push the probabilities toward 0 or 1
# around the few positive rows, while TabICL spreads them more smoothly. The
# next notebooks check what this costs on real card frauds.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * drew `predict_proba` over the feature plane with `DecisionBoundaryDisplay`
#   and a diverging colormap, where white means 0.5 and darker colours mean a
#   more sure model;
# * recognized the inductive bias of each model, a straight line for
#   logistic regression, boxes aligned with the axes for HGBT, and smooth
#   curves for TabICL;
# * saw each model extend its own kind of relationship far from the data;
# * saw that a sure model is not necessarily right, and checked the
#   probabilities with the log loss over 25 shared splits;
# * saw that with a rare class, HGBT can be sure around a few positive rows,
#   while TabICL spreads the probability more smoothly.
