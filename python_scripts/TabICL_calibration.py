# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Can we act on a fraud probability?
#
# A bank rarely answers a suspicious card payment with a plain yes or no. A
# transaction with a 2 percent chance of fraud might trigger a PIN request. At
# 20 percent someone calls the customer. Above 50 percent the payment is
# blocked. Such a policy only works if the probabilities mean what they say:
# among the payments scored at 2 percent, about 2 in 100 should really be
# fraudulent.
#
# In the previous notebook we saw that TabICL's prediction bands kept their
# promised coverage while gradient boosting's did not. Here we ask the same
# question of class probabilities. We cross-validate
# `HistGradientBoostingClassifier` and TabICL on small sets of card
# transactions. Their PR and ROC curves turn out close, so both models rank
# payments by risk about equally well. Their probabilities are another story.
# Gradient boosting labels thousands of payments as one-in-a-million risks
# when a few in a thousand of them are frauds. We then trace that gap back to
# how each model met the log loss.

# %% [markdown]
# ## Learning objectives
#
# - Cross-validate two classifiers with skore and read the spread of each
#   metric across folds.
# - Compare them on a highly imbalanced problem with PR and ROC curves.
# - Read a reliability diagram on log scales, where most predictions sit
#   close to zero.
# - Translate calibration into consequences for a tiered decision policy.
# - Explain why TabICL is well calibrated out of the box while gradient
#   boosting, which also minimizes the log loss, is not.

# %% [markdown]
# ## Fraud is rare
#
# We use the [credit card fraud
# dataset](https://www.openml.org/d/1597), with about 285,000 transactions
# made by European cardholders over two days. The features `V1` to `V28` are
# anonymized principal components, and `Amount` is the transaction amount.

# %%
from sklearn.datasets import fetch_openml

credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas").frame
credit_card["Class"].value_counts(normalize=True)

# %% [markdown]
# Only 0.17 percent of the transactions are fraudulent. A few thousand rows
# picked at random would contain a handful of frauds, too few to learn from or
# to evaluate on. We keep all 492 frauds and a random sample of legitimate
# transactions, for 16,000 rows in total. Fraud stays rare, about 3 percent,
# but each split now holds enough of it. The probabilities we get are
# calibrated for this 3 percent base rate, not for the original one.
#
# With such an imbalance, accuracy is useless. A model that approves
# everything is 97 percent accurate and catches no fraud. We look at ranking
# with PR and ROC curves, and at the probabilities themselves with
# calibration.

# %%
import pandas as pd

is_fraud = credit_card["Class"] == "1"
subset = pd.concat([credit_card[is_fraud], credit_card[~is_fraud].sample(15_508, random_state=0)])
data = subset.drop(columns="Class")
target = (subset["Class"] == "1").astype(int)

# %% [markdown]
# ## Five small training sets, five disjoint test sets
#
# We want small training sets, which TabICL processes quickly on a CPU as
# its context. A usual 5-fold cross-validation would train on 80 percent of
# the rows instead. So we cut the data into 10 stratified folds of 1,600 rows,
# about 49 of them fraudulent, and pair them up: each model trains on one fold
# and is evaluated on the next. Every row is used exactly once, and no test
# row appears in two splits.

# %%
from sklearn.model_selection import StratifiedKFold

folds = [test for _, test in StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(data, target)]
splits = list(zip(folds[::2], folds[1::2]))

# %% [markdown]
# A `CrossValidationReport` from skore fits an estimator on each split and
# caches its predictions. A `ComparisonReport` puts several of them side by
# side. Both models use default hyperparameters, except for TabICL's
# ensemble, which we trim to a single member for speed as in the previous
# notebook.
#
# We also turn on TabICL's `kv_cache`. As in an LLM, the attention keys and
# values of the context are then computed once, during `fit`, instead of at
# every call to `predict_proba`. skore calls the model several times per
# split, so this saves time without changing the predictions.

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
# ## Both models rank transactions about equally well
#
# The summary gives the mean and standard deviation of each metric over the
# five test folds.

# %%
report.metrics.summarize(metric=["roc_auc", "precision", "recall", "log_loss", "brier_score"]).frame()

# %% [markdown]
# With five test folds of about 49 frauds each, the standard deviations are
# rough, but they tell us which gaps to trust. TabICL's ROC AUC is higher by
# about one and a half standard deviations, a modest edge. Precision and
# recall use the default threshold of 0.5. Their differences are smaller than
# the spread across folds, so at this threshold the two models flag almost
# the same payments.
#
# The log loss tells a different story. Gradient boosting's is almost twice
# TabICL's, far beyond the spread. The Brier score barely separates them,
# because it caps the cost of a confident mistake at 1, while the log loss
# grows without limit as the probability given to a fraud approaches zero.
# We come back to this gap below.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

report.metrics.roc().plot()
plt.show()

# %%
report.metrics.precision_recall().plot()
plt.show()

# %% [markdown]
# Each thin line is one fold. The ROC curves hug the top left corner. With
# about 1,550 legitimate transactions per test fold, even dozens of false
# alarms barely move the false positive rate, so ROC curves look flattering
# on imbalanced data.
#
# The PR curves are more demanding, since precision counts false alarms
# against the frauds. A random ranking would give a flat line at the 3 percent
# fraud rate. Both models instead keep a precision close to 1 up to a recall
# of about 0.8. In other words, most frauds in this dataset are easy to spot
# and both models put them at the very top. The remaining fifth is where
# precision collapses and where the folds disagree the most. The average
# precisions overlap well within their spread.
#
# These curves only depend on the order of the scores. Any increasing
# transformation of the probabilities would draw the same curves. A tiered
# policy, however, reads the values.

# %% [markdown]
# ## The probabilities tell them apart
#
# A reliability diagram with linear axes would squeeze almost every
# transaction into its bottom left corner, since most predicted probabilities
# are close to zero. We use log scales on both axes instead, and quantile
# bins, so each point averages 320 test transactions of one fold. With only
# about 49 frauds per fold, some low-risk bins contain no fraud at all. Their
# observed rate is zero, so they fall off the bottom of the plot.
#
# skore returns the cached probabilities of the positive class for each
# split. They are plain NumPy arrays, so scikit-learn's `CalibrationDisplay`
# takes them directly. As for the ROC and PR curves, we draw one line per
# fold.

# %%
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import log_loss

fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharex=True, sharey=True, layout="constrained")
for ax, (name, cv_report) in zip(axes, report.reports_.items()):
    fold_proba = cv_report.get_predictions(data_source="test", response_method="predict_proba")
    losses = []
    for (_, test), p in zip(splits, fold_proba):
        CalibrationDisplay.from_predictions(
            target.iloc[test], p, n_bins=5, strategy="quantile", ax=ax, color="tab:blue", alpha=0.5
        )
        losses.append(log_loss(target.iloc[test], p))
    ax.legend(
        [ax.lines[-1]],
        [f"log loss={np.mean(losses):.3f}±{np.std(losses, ddof=1):.3f}"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    ax.set(xscale="log", yscale="log", xlabel="mean predicted probability", ylabel="observed fraud rate")
    ax.set_title(f"estimator = {name}")
fig.suptitle("Calibration Curve\nPositive label: 1\nData source: Test set")
plt.show()

# %% [markdown]
# The dotted diagonal marks perfect calibration. The legends repeat the test
# log loss from the summary table, as the ROC and PR legends do for their
# own metric. Gradient boosting predicts a
# few in a million for the low-risk bins of every fold. Among these, the
# observed fraud rate is a few in a thousand. Its low probabilities are
# overconfident by about three orders of magnitude, and the folds agree on
# it. TabICL's points sit much closer to the diagonal. They are still above
# it, so TabICL also underestimates the risk of its safest-looking
# transactions, but by a factor of a few rather than a thousand.
#
# No curve gets near the top right corner. The last quantile bin holds the
# riskiest 20 percent of a fold, but only 3 percent of the transactions are
# frauds. That bin mixes nearly all the frauds with many more legitimate
# payments, so both its mean prediction and its fraud rate stay around 0.15.
# The table below looks at the high-risk end directly.
#
# Now we apply the tiered policy from the introduction. The test folds do not
# overlap, so we pool their predictions into a single set of 8,000
# transactions, each predicted by a model that never saw it. For each action,
# we count how many test transactions each model sends there and compare the
# observed fraud rate with the mean predicted probability.

# %%
proba = {
    name: np.concatenate(cv_report.get_predictions(data_source="test", response_method="predict_proba"))
    for name, cv_report in report.reports_.items()
}
fraud = np.concatenate([target.iloc[test] for _, test in splits])

# %%
thresholds = [0, 0.01, 0.1, 0.5, 1]
actions = ["approve", "ask for PIN", "call customer", "block"]

tiers = {
    name: pd.DataFrame(
        {"action": pd.cut(p, thresholds, labels=actions, include_lowest=True), "fraud": fraud, "proba": p}
    )
    .groupby("action", observed=False)
    .agg(transactions=("fraud", "size"), frauds=("fraud", "sum"),
         observed_rate=("fraud", "mean"), predicted_rate=("proba", "mean"))
    for name, p in proba.items()
}
pd.concat(tiers, names=["model"]).round(4)

# %% [markdown]
# Gradient boosting's probabilities are almost binary. It sends nearly every
# transaction either to "approve" or to "block", and uses the two middle
# tiers for only 25 payments. Its "approve" tier contains 38 frauds, about 1
# in 200 payments, while the model predicts essentially zero risk for all of
# them. The bank cannot tell these payments apart from the truly safe ones.
#
# TabICL fills the middle tiers. It asks for a PIN on about 420 payments,
# and the fraud rate among them, 3.1 percent, is close to its prediction of
# 2.6 percent. The same holds for the "call customer" tier. These are the
# payments where a cheap check is worth its cost. Its "approve" tier still
# lets 22 frauds through, which is the underestimation we saw on the
# reliability diagram.

# %% [markdown]
# ## Why the pretrained model is the better calibrated one
#
# Both models are trained with the log loss. It is a proper scoring rule,
# that is, the log loss is lowest on average when the predicted probabilities
# equal the true ones. So the loss alone cannot explain the gap. As with the
# pinball loss in the previous notebook, what matters is which rows the loss
# is computed on.
#
# Gradient boosting minimizes the log loss on its 1,600 training rows. With
# only about 49 frauds, it quickly finds splits that isolate them. Each of the
# 100 boosting iterations then pushes the training probabilities further
# toward 0 and 1, because that keeps lowering the training loss. Early
# stopping would interrupt this, but it is only enabled by default above
# 10,000 samples. skore can compute the log loss on the training folds too.

# %%
report.reports_["HGBT"].metrics.summarize(data_source="train", metric="log_loss").frame()

# %% [markdown]
# On the rows it was fitted on, gradient boosting's log loss is close to
# zero. On the test folds it was about 0.06, twice TabICL's. The confident
# habits it learned on its training rows carry over to new rows, where they
# cost a lot every time a fraud gets a probability near zero.
#
# TabICL was also trained with the log loss, the same cross-entropy that
# teaches an LLM to predict the next token. That training happened once,
# during pretraining, on millions of synthetic tables. Each table was split
# into a context and a set of held-out rows. The model saw only the context
# and was scored on the held-out rows. An overconfident probability on a row
# the model had never seen was costly, so the pretraining rewarded honest
# probabilities on new rows. That is exactly what a tiered fraud policy
# needs.
#
# Calling `fit` on our data changes none of TabICL's weights. It only stores
# the 1,600 transactions as the context. The calibration comes with the
# pretrained model, so we do not need to fit a separate calibration step on
# extra held-out data.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * cross-validated `HistGradientBoostingClassifier` and TabICL with a skore
#   `ComparisonReport`, on small training folds where fraud is rare and
#   accuracy is uninformative;
# * saw that their PR and ROC curves are close, because these curves only
#   depend on how the models rank transactions;
# * passed TabICL's per-fold test probabilities directly to scikit-learn's
#   `CalibrationDisplay.from_predictions`;
# * read a reliability diagram on log scales and found gradient boosting
#   overconfident by orders of magnitude on low-risk transactions;
# * mapped probabilities to actions and saw that only TabICL fills the
#   intermediate tiers with trustworthy probabilities;
# * traced the gap to the rows each model is scored on: gradient boosting
#   drives its log loss to zero on its own training rows, while TabICL was
#   pretrained with the log loss on rows held out of its context.
