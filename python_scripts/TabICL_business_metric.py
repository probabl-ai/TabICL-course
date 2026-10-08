# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # How much money does a fraud detector make?
#
# A bank does not deploy a ROC curve. It deploys a rule that blocks some
# payments and lets the others through, and every decision has a price. A
# blocked fraud saves money. An accepted fraud loses the whole amount. A
# blocked legitimate payment annoys a customer, and an accepted one earns a
# small fee. Most of these prices depend on the amount of each payment, so
# missing a 2,000 € fraud is far worse than missing a 2 € one.
#
# In the previous notebook, the two models ranked transactions about equally
# well. Their probabilities were another story. Gradient boosting called
# thousands of payments one-in-a-million risks when a few in a thousand were
# frauds, and its log loss was almost twice TabICL's. A well calibrated model
# sounds like a better business partner. Does it actually earn more money?
#
# We answer this question with a business metric that counts the euros won
# and lost on each payment. We draw how much each model earns for every
# possible decision threshold, in the style of skore's ROC curves. The answer
# is more subtle than the log loss suggests. A decision only depends on which
# side of the threshold a probability falls. A gap between 0.0001 percent and
# 0.1 percent costs a lot of log loss but changes no decision. Along the way,
# we see where calibration does help, and how easily a few large frauds can
# fool us when we compare models or tune a threshold.

# %% [markdown]
# ## Learning objectives
#
# - Write a business metric whose gains depend on the amount of each
#   transaction, and add it to skore reports.
# - Plot the gain across decision thresholds, with one line per
#   cross-validation fold.
# - Relate the shape of each gain curve to how calibrated the model's
#   probabilities are.
# - Explain why a large gap in log loss can translate into a small gap in
#   money.
# - Choose a threshold from the costs alone or from held-out folds, and
#   compare both with the default threshold.
# - Compare two models fold by fold when the spread across folds is large.

# %% [markdown]
# We start from the setup of the previous notebook. We keep the 492 frauds of
# the [credit card fraud dataset](https://www.openml.org/d/1597) and a random
# sample of legitimate transactions, for 16,000 rows. We cut them into 10
# stratified folds of 1,600 rows and pair them up. Each model is trained on
# one fold and evaluated on the next.

# %%
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold

credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas").frame
is_fraud = credit_card["Class"] == "1"
subset = pd.concat([credit_card[is_fraud], credit_card[~is_fraud].sample(15_508, random_state=0)])
data = subset.drop(columns="Class")
target = (subset["Class"] == "1").astype(int)

folds = [test for _, test in StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(data, target)]
splits = list(zip(folds[::2], folds[1::2]))

# %% [markdown]
# As before, a `ComparisonReport` gathers one `CrossValidationReport` per
# model. Fitting TabICL on the five splits takes most of the running time of
# this notebook.

# %%
# %pip install -q tabicl skore
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import ComparisonReport, CrossValidationReport
from tabicl import TabICLClassifier

models = {
    "HGBT": HistGradientBoostingClassifier(random_state=0),
    "TabICL": TabICLClassifier(n_estimators=1, kv_cache=True, random_state=0),
}
report = ComparisonReport(
    {
        name: CrossValidationReport(model, data, target, splitter=splits, pos_label=1)
        for name, model in models.items()
    }
)

# %% [markdown]
# ## Each decision has a price
#
# We borrow the gains and losses from the [cost-sensitive learning
# example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
# of scikit-learn, which uses the same dataset:
#
# - accepting a legitimate payment earns a fee of 2 percent of its amount;
# - blocking a legitimate payment costs 5 €, for example in customer
#   service;
# - accepting a fraud loses its whole amount;
# - blocking a fraud earns 50 €, a rough value for the trouble it avoids.
#
# Blocking a payment is the positive prediction. The sum of these gains over
# a test fold is our business metric. Unlike precision or recall, it cannot
# be computed from the confusion matrix alone, because two of the four gains
# depend on the amount of each transaction.

# %%
def business_gain(y_true, y_pred, amount):
    fraud = y_true == 1
    blocked = y_pred == 1
    return (
        0.02 * amount[~fraud & ~blocked].sum()
        - 5 * (~fraud & blocked).sum()
        - amount[fraud & ~blocked].sum()
        + 50 * (fraud & blocked).sum()
    )


# %% [markdown]
# Two simple policies frame what a model can achieve. The first accepts every
# payment. The second is a perfect detector that blocks exactly the frauds.

# %%
import numpy as np

amount = data["Amount"].to_numpy()
fold_targets = [target.iloc[test].to_numpy() for _, test in splits]
fold_amounts = [amount[test] for _, test in splits]

pd.DataFrame(
    {
        "accept everything": [business_gain(y, np.zeros_like(y), a) for y, a in zip(fold_targets, fold_amounts)],
        "perfect detector": [business_gain(y, y, a) for y, a in zip(fold_targets, fold_amounts)],
    }
).agg(["mean", "std"]).round()

# %% [markdown]
# Accepting everything loses money on every fold. This is a consequence of our
# subsampling. Frauds make up 3 percent of our rows, against 0.17 percent in
# the full dataset, where the fees on legitimate payments outweigh the frauds.
# All the euros in this notebook are therefore per fold of our rebalanced
# data, and they do not estimate what a bank would earn. They remain a fair
# basis to compare models and thresholds with each other.
#
# We now add the business metric to the comparison report, which passes it on
# to each cross-validation report. skore has already cached the predictions
# of each split. A scorer built with `make_scorer` and
# `response_method="predict"` reuses them, so TabICL does not need to predict
# again. Such a scorer only receives the true and predicted labels. The true
# labels keep the index of `data`, so we use it to look up the amount of each
# transaction.
#
# `predict` blocks a payment when its predicted probability of fraud exceeds
# 0.5.

# %%
from sklearn.metrics import make_scorer


def business_gain_score(y_true, y_pred):
    return business_gain(y_true.to_numpy(), y_pred, data.loc[y_true.index, "Amount"].to_numpy())


report.metrics.add(make_scorer(business_gain_score, response_method="predict"), name="business_gain")
report.metrics.summarize(metric=["business_gain", "precision", "recall"]).frame()

# %% [markdown]
# Both models earn money. They close about 70 percent of the gap between
# accepting everything and the perfect detector. TabICL earns about 350 € more
# on average. This gap is small compared to the standard deviations across
# folds, which lie between 1,100 € and 1,500 €. We come back to this spread at
# the end of the notebook.

# %% [markdown]
# ## The gain across decision thresholds
#
# The threshold of 0.5 is a convention of `predict`, not a business decision.
# A lower threshold blocks more frauds and more legitimate payments. A higher
# one does the opposite. A ROC curve shows this trade-off in terms of error
# rates. We can draw the same kind of curve with euros on the vertical axis.
#
# skore caches the predicted probabilities of each test fold, so we only need
# to compute the business gain for a grid of thresholds. Both models give
# many transactions a probability far below 1 percent, so we space the
# thresholds on a log scale.

# %%
thresholds = np.geomspace(1e-4, 0.9, 80)

proba = {
    name: cv_report.get_predictions(data_source="test", response_method="predict_proba")
    for name, cv_report in report.reports_.items()
}
gains = {
    name: np.array(
        [
            [business_gain(y, p >= threshold, a) for threshold in thresholds]
            for y, a, p in zip(fold_targets, fold_amounts, fold_proba)
        ]
    )
    for name, fold_proba in proba.items()
}

# %% [markdown]
# We mimic the layout of skore's ROC display, with one panel per model, one
# thin line per fold, and a legend below each panel. The legend gives the highest
# gain of the mean curve, with the standard deviation across folds at that
# threshold. The dotted line marks the default threshold of 0.5.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharex=True, sharey=True, layout="constrained")
for ax, (name, fold_gains) in zip(axes, gains.items()):
    best = fold_gains.mean(axis=0).argmax()
    for g in fold_gains:
        ax.plot(thresholds, g, color="tab:blue", alpha=0.4)
    ax.axvline(0.5, color="black", linestyle=":")
    ax.legend(
        [ax.lines[0], ax.lines[-1]],
        [
            (
                f"best gain={fold_gains[:, best].mean():.0f}±{fold_gains[:, best].std(ddof=1):.0f} € "
                f"at threshold {thresholds[best]:.3f}"
            ),
            "default threshold (0.5)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    ax.set(xscale="log", xlabel="decision threshold", ylabel="business gain (€)")
    ax.set_title(f"estimator = {name}")
fig.suptitle("Business Gain Curve\nPositive label: 1\nData source: Test set")
plt.show()

# %% [markdown]
# The two models draw very different curves. Each shape reflects what we saw
# on the reliability diagrams of the previous notebook.
#
# Gradient boosting's curves are almost flat over four orders of magnitude.
# Its probabilities are nearly binary, with legitimate payments close to 0
# and frauds close to 1. Only about 3 percent of the test transactions get a
# probability between 0.0001 and 0.5. Moving the threshold within this range
# therefore changes few decisions. The scikit-learn example linked above shows
# a similar plateau when a threshold is tuned on the rows the model was
# trained on, and calls it a symptom of overfitting. Our curves are computed
# on test folds. The plateau still appears, because the overconfidence that
# gradient boosting learns on its training rows carries over to new rows.
#
# TabICL's curves have a plateau above about 1 percent and fall steeply
# below it. TabICL gives legitimate payments small but honest risks. Half of
# them get a probability above 0.08 percent, and one in ten above 0.5
# percent. A threshold below 1 percent blocks thousands of these payments, at
# 5 € each. On its plateau, TabICL earns slightly more than gradient boosting
# in most folds.

# %% [markdown]
# ## Calibration matters only near the decision threshold
#
# Where should the threshold be? Take a payment of 20 €. Blocking it by
# mistake costs 5 € and the 0.40 € fee, so 5.40 €. Blocking it when it is a
# fraud earns 50 € and avoids losing the 20 €, so 70 € in total. Blocking
# pays off once the risk of fraud exceeds 5.40 € out of 75.40 €, that is,
# about 7 percent. For a payment of 1,000 €, the same reasoning gives about
# 2 percent. Below 2 percent, the bank should accept a payment whatever its
# amount.
#
# This zone below 2 percent is where gradient boosting's calibration was
# worst. We pool the test folds and look at the transactions each model
# places there.

# %%
fraud = np.concatenate(fold_targets)
low_risk = {}
for name, fold_proba in proba.items():
    p = np.concatenate(fold_proba)
    accepted = p < 0.02
    low_risk[name] = {
        "transactions": accepted.sum(),
        "frauds": fraud[accepted].sum(),
        "observed fraud rate": fraud[accepted].mean(),
        "mean predicted probability": p[accepted].mean(),
    }
pd.DataFrame.from_dict(low_risk, orient="index")

# %% [markdown]
# Gradient boosting underestimates the fraud rate of this zone by a factor of
# about 100, and TabICL by a factor of about 2. The log loss charges a lot for
# each fraud predicted at 0.001 percent instead of 0.1 percent. The bank does
# not care, because both predictions lead to the same decision. The fraud is
# accepted either way, and the bank loses the same amount.
#
# What costs money is the number of frauds in this zone, 39 for gradient
# boosting and 29 for TabICL. This count depends on how well each model ranks
# frauds above legitimate payments, not on calibration. This is why the large
# gap in log loss turns into a small gap in money.

# %% [markdown]
# ## Three ways to choose the threshold
#
# Calibration does help in one place. With calibrated probabilities, we can
# set the threshold from the costs alone, without any data. We apply the
# reasoning of the 20 € payment to the amount of each transaction, which
# gives each transaction its own break-even risk. With miscalibrated
# probabilities, such a threshold has no reason to be a good one.
#
# The usual alternative is to tune the threshold on data. The best gains in
# the legends above are optimistic, because we chose each threshold on the
# very folds we report the gain on. With about 49 frauds per fold, a
# threshold can look best simply because it happens to catch one large fraud.
#
# `TunedThresholdClassifierCV` from scikit-learn avoids this bias. It chooses
# the threshold with an internal cross-validation on the training data. Here
# it would refit TabICL five times within each split, and its validation
# folds would hold only about 10 frauds each. Since we already have the gain
# curves of the five test folds, we apply the same idea at no cost. For each
# fold, we pick the threshold that maximizes the mean gain of the four other
# folds. Then we report the gain at this threshold on the held-out fold.
#
# We compare these two choices with the default threshold of 0.5.

# %%
threshold_choices = {}
for name, fold_proba in proba.items():
    breakeven = [(5 + 0.02 * a) / (55 + 1.02 * a) for a in fold_amounts]
    g = gains[name]
    threshold_choices[(name, "default")] = [
        business_gain(y, p >= 0.5, a) for y, a, p in zip(fold_targets, fold_amounts, fold_proba)
    ]
    threshold_choices[(name, "from costs")] = [
        business_gain(y, p >= b, a) for y, a, p, b in zip(fold_targets, fold_amounts, fold_proba, breakeven)
    ]
    threshold_choices[(name, "tuned on other folds")] = [
        g[k, np.delete(g, k, axis=0).mean(axis=0).argmax()] for k in range(len(g))
    ]
pd.DataFrame(threshold_choices).agg(["mean", "std"]).T.round()

# %% [markdown]
# For TabICL, the threshold derived from the costs earns about as much as the
# default one, without looking at any data. Tuning on the other folds even
# lowers its gain by about 300 €. For gradient boosting, both alternatives
# raise the gain by about 360 € compared with the default. Its probabilities are not calibrated,
# but its curves are flat between 0.0001 and 0.3, so almost any threshold
# below the default does as well.
#
# All these differences are well within the spread across folds. With so few
# frauds per fold, the position of the best threshold is mostly noise, and
# tuning it pays off only with many more frauds to tune it on. A broad
# plateau is therefore a reassuring sign. It means that the gain does not
# hinge on the exact threshold.
# %% [markdown]
# ## A few large frauds decide the ranking
#
# The standard deviation across folds is about half of the mean gain. Most of
# this spread comes from frauds that slip through. We look at the largest
# fraud that each model accepts in each test fold, at the default threshold.

# %%
pd.DataFrame(
    {
        name: [a[(y == 1) & (p < 0.5)].max() for y, a, p in zip(fold_targets, fold_amounts, fold_proba)]
        for name, fold_proba in proba.items()
    }
).round()

# %% [markdown]
# In the first fold, both models accept the same fraud of about 2,100 €. At a
# 2 percent fee, the bank needs about 106,000 € of legitimate payments to
# earn this money back. That is most of the legitimate volume of a test fold.
# A single transaction can thus move the gain of a fold by thousands of
# euros.
#
# Such a transaction affects both models in the same way, because both are
# evaluated on the same test folds. Comparing the gains fold by fold removes
# this shared part of the spread. skore gives the value of each split when we
# ask for no aggregation.

# %%
per_fold = (
    report.metrics.summarize(metric="business_gain")
    .frame(aggregate=None, flat_index=False)
    .loc["business_gain"]
    .unstack("estimator")
)
per_fold["TabICL - HGBT"] = per_fold["TabICL"] - per_fold["HGBT"]
per_fold.round()

# %% [markdown]
# The gains of each model range from about 1,100 € to 4,600 €. The gaps
# between the two models range from about -60 € to 1,400 €. TabICL earns more
# in three folds, ties in one, and loses a few dozen euros in the last. Five
# folds remain too few to conclude that TabICL is reliably more profitable.
# Still, the paired view tells us much more than the overlap of two large
# standard deviations.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * wrote a business metric where the gains depend on the amount of each
#   transaction, and added it to a skore `ComparisonReport` without
#   recomputing any prediction;
# * drew the gain of each model across decision thresholds from the cached
#   test probabilities, in the style of skore's displays;
# * related the flat curves of gradient boosting to its nearly binary
#   probabilities, and the cliff of TabICL's curves to the small but honest
#   risks it gives to legitimate payments;
# * saw that gradient boosting's miscalibration sits below any sensible
#   threshold, where it costs log loss but no money;
# * set a threshold from the costs alone, which calibrated probabilities make
#   possible, and another one on held-out folds, and found that neither
#   reliably beats the default with about 49 frauds per fold;
# * found that a handful of large frauds drive most of the spread across
#   folds, and compared the two models fold by fold to remove this shared
#   noise.
