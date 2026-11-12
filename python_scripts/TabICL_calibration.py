# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Can we act on a fraud probability?
#
# For each card payment, a bank decides whether to accept it or block it. Each
# decision has a price. An accepted fraud loses its whole amount, while a
# blocked legitimate payment costs customer support. If the probabilities
# returned by `predict_proba` match the observed fraud rate, the bank can turn
# these prices into a decision rule. Among all the payments scored at 2 percent,
# about 2 in 100 should then be fraudulent. Richer policies, such as asking for
# a PIN at intermediate risks, need this match just as much.
#
# In this notebook, we cross-validate `HistGradientBoostingClassifier` and
# TabICL on small sets of card transactions to check which one gets the closer
# match, much as TabICL's prediction intervals came closer to their promised
# coverage in the notebook on the pinball loss. By the end, you should be able
# to:
#
# - Cross-validate two classifiers with skore and read the spread of each metric
#   across splits.
# - Read the decisions of a classifier at its default threshold with a confusion
#   matrix, precision and recall.
# - Compare two classifiers on a highly imbalanced problem with ROC and PR
#   curves.
# - Read a reliability diagram on log scales, where most predictions sit close
#   to zero.
# - Derive a decision rule from the costs of each decision, and measure in euros
#   what calibrated probabilities are worth.
# - Explain why TabICL tends to output better calibrated probabilities out of
#   the box than gradient boosting.

# %% [markdown]
# ## Data and cross-validation
#
# We use the [credit card fraud
# dataset](https://www.openml.org/d/1597), with about 285,000 transactions
# made by European cardholders over two days. The features `V1` to `V28` are
# anonymized principal components, and `Amount` is the transaction amount.
#
# For pedagogical purposes, we keep all 492 frauds and a random sample of
# legitimate transactions, for 10,000 rows in total. Fraud stays rare,
# representing about 5 percent of all transactions.

# %%
import pandas as pd
from sklearn.datasets import fetch_openml

credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas").frame

is_fraud = credit_card["Class"] == "1"
subset = pd.concat(
    [credit_card[is_fraud], credit_card[~is_fraud].sample(9_508, random_state=0)]
)

data = subset.drop(columns="Class")
target = (subset["Class"] == "1").astype(int)
target.value_counts(normalize=True)

# %% [markdown]
# A usual 10-fold cross-validation would train on 9,000 rows, too slow for
# TabICL on a laptop CPU. Instead, StratifiedShuffleSplit draws 10 random
# splits, each with a smaller training set and test set. Notice that with
# (Stratified)ShuffleSplit each split is drawn independently, so the same row
# can appear in several splits, for instance in two different test sets.
# Stratification just ensures that all sets keep the 5 percent fraud rate.
#
# If you have more compute, such as a GPU, try raising `train_size`, `test_size`
# or `n_splits`. A larger training set gives TabICL a richer context, larger
# test sets make each estimate less noisy, and more splits make their mean more
# precise.

# %%
from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(
    n_splits=10, train_size=1_500, test_size=500, random_state=0
)  # about 74 frauds per training set and 25 per test set

# %% [markdown]
# ```{note}
# Stratification ensures that rare classes are represented in every split.
# If a class is absent from a split, some classification metrics, such as
# the ROC AUC, are undefined.
#
# However, stratification makes the splits more homogeneous than new data
# would be. With a severe class imbalance, this can artificially reduce the
# variability of the metrics across splits, especially for metrics that
# depend on the proportion of the positive class, such as the average
# precision (AP). As a consequence, the observed spread underestimates the
# uncertainty of the model's performance on new data.
# ```

# %% [markdown]
# A `CrossValidationReport` from skore fits an estimator on each split and
# caches its predictions. A `ComparisonReport` puts several of them side by
# side. Both models use default hyperparameters, except for TabICL's
# ensemble, which we trim to a single member for speed as in the previous
# notebook.
#
# We also turn on TabICL's `kv_cache`. As seen in a previous exercise, the
# cache moves the work on the training rows into `fit` and pays off once the
# same fitted model predicts more than once. skore calls the model several
# times per split, so this saves time without changing the predictions.

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
        name: CrossValidationReport(model, data, target, splitter=cv, pos_label=1)
        for name, model in models.items()
    }
)

# %% [markdown]
# ## Decisions at the default threshold
#
# `predict` blocks a payment when its probability of fraud is at least 0.5.
# The confusion matrix counts the test payments by true and predicted class.
# skore draws one matrix per model, with the mean count over the ten test
# sets and its standard deviation.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

fig = report.metrics.confusion_matrix().plot()
for ax in fig.axes:
    ax.set(xticklabels=["legitimate", "fraud"], yticklabels=["legitimate", "fraud"])
plt.show()


# %% [markdown]
# Out of the 25 frauds of a test set, both models catch about 21 and miss
# about 4. They block fewer than one legitimate payment per test set.
# Precision and recall summarize these counts. Precision is the share of
# blocked payments that are frauds, and recall is the share of frauds that
# get blocked.

# %%
report.metrics.summarize(metric=["precision", "recall"]).frame()

# %% [markdown]
# With ten test sets of about 25 frauds each, the standard deviations are
# rough, but they tell us which gaps to trust. The differences in precision
# and recall are smaller than the spread across splits, so at this threshold
# the two models flag almost the same payments.

# %% [markdown]
# ## Ranking with ROC and PR curves
#
# A single threshold hides most of what a model predicts. ROC and PR curves
# sweep all thresholds. Each thin line is one split.

# %%
report.metrics.roc().plot()
plt.show()

# %%
report.metrics.precision_recall().plot()
plt.show()

# %% [markdown]
# The ROC curves hug the top left corner. With about 475 legitimate
# transactions per test set, even dozens of false alarms barely move the
# false positive rate, so ROC curves look flattering on imbalanced data. The
# areas under the curves, in the legends, are nearly equal for both models.
#
# The PR curves are more demanding, since precision counts false alarms
# against the frauds. A random ranking would give a flat line at the 5
# percent fraud rate. Both models instead keep a precision close to 1 up to a
# recall of about 0.8. In other words, most frauds in this dataset are easy to
# spot and both models put them at the very top. The remaining fifth is where
# precision collapses and where the splits disagree the most. The average
# precisions overlap well within their spread.
#
# These curves only depend on the order of the scores. Any increasing
# transformation of the probabilities would draw the same curves. A decision
# rule based on costs, however, reads the values.
#
# ```{note}
# Frauds are about 30 times rarer in production than in our sample. The ROC
# curve does not depend on the fraud rate, because each of its two axes is
# computed within one class. Precision compares frauds with false alarms, so
# the PR curve would drop in production. The probabilities would be too high
# as well, since they follow the fraud rate of the training data. Before
# deploying a model, check its probabilities on data with the production
# fraud rate.
# ```

# %% [markdown]
# ## Calibration of the probabilities
#
# Two metrics read the probabilities themselves. The log loss and the Brier
# score both reward a probability close to 1 for a fraud and close to 0 for
# a legitimate payment.

# %%
report.metrics.summarize(metric=["log_loss", "brier_score"]).frame()

# %% [markdown]
# Gradient boosting's log loss is almost twice TabICL's, far beyond the
# spread. The Brier score barely separates them, because it caps the cost of
# a confident mistake at 1, while the log loss grows without limit as the
# probability given to a fraud approaches zero.
#
# A model is calibrated when its probabilities match the observed fraud
# rates, as in the 2-in-100 example of the introduction. A reliability
# diagram checks this. It groups the test transactions by predicted
# probability, and plots the mean prediction of each group against its
# observed fraud rate.
#
# A reliability diagram with linear axes would squeeze almost every
# transaction into its bottom left corner, since most predicted probabilities
# are close to zero. We use log scales on both axes instead, and quantile
# bins, so each point averages 100 transactions of one test set. With only
# about 25 frauds per test set, a low-risk bin holds zero, one or two frauds,
# so its observed rate is 0, 1 or 2 percent. Bins without any fraud have an
# observed rate of zero, so they fall off the bottom of the plot.
#
# skore returns the cached probabilities of the positive class for each
# split. They are plain NumPy arrays, so scikit-learn's `CalibrationDisplay`
# takes them directly. As for the ROC and PR curves, we draw one line per
# split.

# %%
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import log_loss

limits = [1e-6, 1e0]
fig, axes = plt.subplots(
    1, 2, figsize=(12, 6.75), sharex=True, sharey=True, layout="constrained"
)
for ax, (name, cv_report) in zip(axes, report.reports_.items()):
    fold_proba = cv_report.get_predictions(
        data_source="test", response_method="predict_proba"
    )
    losses = []
    for split_report, p in zip(cv_report.reports_, fold_proba):
        CalibrationDisplay.from_predictions(
            split_report.y_test,
            p,
            n_bins=5,
            strategy="quantile",
            ax=ax,
            color="tab:blue",
            alpha=0.5,
            ref_line=False,
        )
        losses.append(log_loss(split_report.y_test, p))
    ax.plot(limits, limits, color="k", linestyle="--")
    ax.legend(
        [ax.lines[0], ax.lines[-1]],
        [
            f"log loss={np.mean(losses):.2f}±{np.std(losses, ddof=1):.2f}",
            "Perfect calibration",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    ax.set_box_aspect(1)
    ax.set(
        xlim=limits,
        ylim=limits,
        xscale="log",
        yscale="log",
        xlabel="mean predicted probability",
        ylabel="observed fraud rate",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"estimator = {name}")
fig.suptitle("Calibration Curve\nPositive label: 1\nData source: Test set")
plt.show()

# %% [markdown]
# The dashed diagonal marks perfect calibration. The legends repeat the test
# log loss from the summary table, as the ROC and PR legends do for their
# own metric. Gradient boosting predicts a few in a million for the low-risk
# bins of every split. The low-risk bins that contain a fraud have an
# observed rate of 1 or 2 percent, about three orders of magnitude higher.
# The bins without any fraud are missing from the plot, so this overstates
# the gap, but every split points the same way. TabICL's points sit much
# closer to the diagonal, though still above it. TabICL also underestimates
# the risk of its safest-looking transactions, but far less.
#
# No curve gets near the top right corner. The last quantile bin holds the
# riskiest 20 percent of a test set, but only 5 percent of the transactions
# are frauds. That bin mixes nearly all the frauds with many more legitimate
# payments, so both its mean prediction and its fraud rate stay around 0.25.

# %% [markdown]
# ## A decision rule from the costs
#
# We now put a price on each decision about a payment:
#
# - accepting a legitimate payment earns a commission of 2 percent of its
#   amount;
# - accepting a fraud loses its whole amount, plus 20 € for the
#   investigation and the customer support;
# - blocking a legitimate payment costs 10 €, for the customer support and
#   the risk that the customer leaves;
# - blocking a fraud neither earns nor costs anything.
#
# The sum of these gains over a test set measures how much money a decision
# policy makes.


# %%
def business_gain(y_true, y_pred, amount):
    fraud = y_true == 1
    blocked = y_pred == 1
    return (
        0.02 * amount[~fraud & ~blocked].sum()
        - (amount[fraud & ~blocked] + 20).sum()
        - 10 * (~fraud & blocked).sum()
    )


# %% [markdown]
# Take a payment of 20 € that a model scores with a probability `p` of fraud.
# If we block it, we lose 10 € when it is legitimate, that is, with
# probability `1 - p`, and nothing otherwise. If we accept it, we lose 40 €
# with probability `p` and earn a commission of 0.40 € otherwise. Blocking
# pays off when
#
# `-10 (1 - p) > -40 p + 0.40 (1 - p)`,
#
# that is, when `p` is above 10.40 / 50.40, about 21 percent. For a payment of
# 1,000 €, the same reasoning gives about 3 percent. In general, a payment of
# amount `a` should be blocked when `p` exceeds
# `(10 + 0.02 a) / (30 + 1.02 a)`. This break-even risk lies between 2 and 33
# percent. It is lower for larger amounts, because accepting a large fraud
# costs more.
#
# This rule needs no data beyond the costs, and it gives each payment its own
# threshold. But it reads `p` as a true probability of fraud, so it is only
# sound for a calibrated model. We compare it with the default threshold of
# 0.5, whose decisions we counted in the confusion matrix. skore gives us the
# test rows of each split, including the `Amount` column.

# %%
results = {}
for name, cv_report in report.reports_.items():
    fold_proba = cv_report.get_predictions(
        data_source="test", response_method="predict_proba"
    )
    rows = []
    for split_report, p in zip(cv_report.reports_, fold_proba):
        y = split_report.y_test.to_numpy()
        a = split_report.X_test["Amount"].to_numpy()
        blocked_by_rule = p >= (10 + 0.02 * a) / (30 + 1.02 * a)
        extra = blocked_by_rule & (p < 0.5)
        rows.append(
            {
                "default (0.5)": business_gain(y, p >= 0.5, a),
                "from the costs": business_gain(y, blocked_by_rule, a),
                "extra frauds blocked": (extra & (y == 1)).sum(),
                "extra legitimate blocked": (extra & (y == 0)).sum(),
            }
        )
    results[name] = pd.DataFrame(rows)

pd.concat(
    {
        name: df[["default (0.5)", "from the costs"]].agg(["mean", "std"])
        for name, df in results.items()
    }
).round()

# %% [markdown]
# For TabICL, the rule from the costs earns about 150 € more per test set
# than the default threshold, with a smaller spread. Most of this gain comes
# from a single test set, where the rule blocks four frauds that TabICL
# scored between 5 and 40 percent, one of them worth 723 €. On the other test
# sets, the rule wins or loses up to about 240 €.
#
# For gradient boosting, the rule earns about 50 € more. It cannot do much
# more. The rule and the default threshold only disagree on payments scored
# between 2 and 50 percent, and gradient boosting puts only about 0.6 percent
# of the payments there, against about 5 percent for TabICL. We count the
# payments that the rule blocks while the default threshold accepts them,
# over the ten test sets.

# %%
pd.DataFrame(
    {
        name: df[["extra frauds blocked", "extra legitimate blocked"]].sum()
        for name, df in results.items()
    }
)

# %% [markdown]
# For TabICL, the rule blocks 8 more frauds, at the price of 25 legitimate
# payments. For gradient boosting, it blocks 6 more frauds, at the price of 3
# legitimate payments. The few payments that gradient boosting scores in
# between are riskier than it predicts, so blocking them pays off, but they
# are rare. Most of the frauds it misses get a probability below 2 percent,
# where the rule accepts any payment, whatever its amount. A calibrated model
# gives the rule more to work with, namely the payments whose risk lies in
# between.
#
# The log loss and the money do not weigh mistakes the same way. The log
# loss charges a lot for a fraud predicted at 0.001 percent instead of 0.1
# percent, which is why gradient boosting's log loss is almost twice
# TabICL's. The bank does not care, as long as both predictions fall on the
# same side of the break-even risk. The fraud is accepted either way, and the
# bank loses the same amount.

# %% [markdown]
# ## Why TabICL is better calibrated
#
# The notebook on the pinball loss announced this question. Both models
# minimize the log loss, and the log loss is lowest when the predicted probabilities equal the
# true ones, so naming the loss explains nothing. As for the prediction
# intervals, what matters is which rows the loss is computed on.
#
# Gradient boosting minimizes the log loss on its 1,500 training rows, which
# hold about 74 frauds. Its trees quickly isolate them, and each of the 100
# boosting iterations pushes the training probabilities closer to 0 or 1,
# since that keeps lowering the training loss. skore can compute the log loss
# on the training sets too.

# %%
report.reports_["HGBT"].metrics.summarize(
    data_source="train", metric="log_loss"
).frame()

# %% [markdown]
# The training log loss is close to zero, against about 0.07 on the test
# sets, so gradient boosting overfits. Its prediction interval in the notebook
# on the pinball loss failed the same way. It was sized for the errors on the training
# rows and came out too short on the test hours. Here, the probabilities are
# sized for training frauds the trees have already isolated, and are too
# confident on new rows.
#
# Overconfidence costs the most when a fraud gets a probability near zero.
# To check that this is where the test log loss comes from, we compute the
# log loss of each test row and keep the 5 most costly rows of each split. We
# count the frauds among them, and how many of those got a probability below
# 0.1 percent.

# %%
rows = []
for name, cv_report in report.reports_.items():
    fold_proba = cv_report.get_predictions(
        data_source="test", response_method="predict_proba"
    )
    for split_report, p in zip(cv_report.reports_, fold_proba):
        y = split_report.y_test.to_numpy()
        row_loss = -np.log(np.where(y == 1, p, 1 - p))
        worst = np.argsort(row_loss)[-5:]
        rows.append(
            {
                "model": name,
                "share of the log loss": row_loss[worst].sum() / row_loss.sum(),
                "frauds": (y[worst] == 1).sum(),
                "frauds below 0.1%": ((y[worst] == 1) & (p[worst] < 0.001)).sum(),
            }
        )

pd.DataFrame(rows).groupby("model").agg(["mean", "std"]).T.round(2)

# %% [markdown]
# For both models, 5 rows out of 500 carry most of the test log loss, and
# about 4 of them are frauds. Both models miss a few frauds, but gradient
# boosting is far more certain about them. On average, 2.6 of its 5 worst
# rows are frauds scored below 0.1 percent, against 0.2 for TabICL.
#
# TabICL's weights were also fitted with the log loss, but during
# pretraining, and only on rows held out of the context, as for the prediction
# intervals. Its probabilities are therefore sized for rows it has not seen,
# which is the situation of our test sets. Our 1,500 rows only serve as
# context. As the learning curve notebook announced, the inductive bias
# matters most where data is scarce, and here the scarce rows are the 74 or so
# training frauds. The previous notebook showed the same contrast on a rare
# class in two dimensions. Where gradient boosting drives the probability of a
# few frauds to almost zero, TabICL stays more cautious.
#
# The notebook on the pinball loss gave a second reason for TabICL's
# prediction intervals, namely that one model produced all 999 quantiles. It
# has no counterpart here, since a classifier outputs one probability per
# class. Also, the calibration is not
# guaranteed. It holds as far as our transactions resemble the pretraining
# tables, and it follows the fraud rate of the context, as the note on the
# production fraud rate explained.
#
# Gradient boosting can also be scored on rows it did not fit. Early stopping
# (`early_stopping=True`, on by default only above 10,000 samples) and a
# random search on the log loss reduce its overfitting, while
# `CalibratedClassifierCV` corrects its probabilities. All three hold out
# rows, which have to come from the same 74 or so frauds. TabICL uses every
# fraud directly as context.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * cross-validated `HistGradientBoostingClassifier` and TabICL with a skore
#   `ComparisonReport`, on small training sets where fraud is rare and
#   accuracy is uninformative;
# * counted their decisions at the default threshold with a confusion
#   matrix, and found them almost identical;
# * saw that their ROC and PR curves are close, because these curves only
#   depend on how the models rank transactions;
# * passed each model's per-split test probabilities directly to
#   scikit-learn's `CalibrationDisplay.from_predictions`;
# * read a reliability diagram on log scales and found gradient boosting
#   overconfident by orders of magnitude on low-risk transactions;
# * derived a decision rule from the costs of each decision, which reads
#   the probabilities as true risks, and saw it pay off for TabICL;
# * traced the gap to the rows each model is scored on, since gradient
#   boosting drives its log loss to zero on its own training rows, while
#   TabICL was pretrained with the log loss on rows held out of its context;
# * found that a few frauds scored near zero carry most of gradient
#   boosting's test log loss;
# * listed ways to score gradient boosting on held-out rows as well, namely
#   early stopping, a random search on the log loss and
#   `CalibratedClassifierCV`.
