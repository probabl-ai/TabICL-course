# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # How much money does a fraud detector make?
#
# In the previous notebook, gradient boosting and TabICL ranked card payments
# by risk about equally well, but only TabICL gave calibrated probabilities.
# We put a price on each decision and derived a rule that turns a calibrated
# probability into a decision without any data. It paid off for TabICL.
#
# What if the probabilities are not calibrated? The usual answer is to tune
# the decision threshold on labelled data, with the money it makes as the
# score. In this notebook, we write this business metric so that
# scikit-learn's tools can use it, draw the gain across thresholds, and tune
# the threshold with `TunedThresholdClassifierCV`. We will see that a tuned
# threshold only depends on how the predictions are ordered, so calibration
# matters less once we tune. With a few dozen frauds per split, however,
# tuning is noisy, and it does not beat the free rule of a calibrated
# model.

# %% [markdown]
# ## Learning objectives
#
# - Write a business metric whose gains depend on the amount of each
#   transaction, and pass the amounts to it with metadata routing.
# - Frame the gains of a model between those of simple policies, such as
#   accepting every payment.
# - Plot the gain across decision thresholds, with one line per
#   cross-validation split, and relate its shape to calibration.
# - Tune a decision threshold on data with `TunedThresholdClassifierCV`, and
#   compare it with the rule from the costs.
# - Explain why tuning a threshold makes calibration matter less, and what
#   calibration still buys when tuning data is scarce.
# - Measure how much the gain depends on the single threshold a bank would
#   deploy.
# - Compare two models split by split when the spread across splits is large.

# %% [markdown]
# We reuse the setup of the previous notebook. We keep the 492 frauds of the
# [credit card fraud dataset](https://www.openml.org/d/1597) and a random
# sample of legitimate transactions, for 10,000 rows. `StratifiedShuffleSplit`
# draws 10 splits, each with 1,500 training rows, about 74 of them frauds,
# and 500 test rows, about 25 of them frauds. We also keep the amount of each
# transaction aside, since our metric needs it.

# %%
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas").frame
is_fraud = credit_card["Class"] == "1"
subset = pd.concat(
    [credit_card[is_fraud], credit_card[~is_fraud].sample(9_508, random_state=0)]
)
data = subset.drop(columns="Class")
target = (subset["Class"] == "1").astype(int)
amount = data["Amount"].to_numpy()

cv = StratifiedShuffleSplit(
    n_splits=10, train_size=1_500, test_size=500, random_state=0
)
# On a GPU, a larger test set gives less noisy estimates:
# cv = StratifiedShuffleSplit(
#     n_splits=10, train_size=1_500, test_size=1_500, random_state=0
# )

# %% [markdown]
# ## Business metric and metadata routing
#
# We use the same gains and losses as in the previous notebook:
#
# - accepting a legitimate payment earns a commission of 2 percent of its
#   amount;
# - accepting a fraud loses its whole amount, plus 20 € for the
#   investigation, the customer support and the damage to the bank's
#   reputation;
# - blocking a legitimate payment costs 10 €, for the customer support and
#   the risk that the customer leaves;
# - blocking a fraud neither earns nor costs anything.
#
# In practice, the data scientist would set the fixed costs with the people
# who run the business. Blocking a payment is the positive prediction. The
# sum of these gains over a test set is our business metric. Unlike
# precision or recall, it cannot be computed from the confusion matrix alone,
# because two of the gains depend on the amount of each transaction.

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
# scikit-learn tools such as `cross_validate` evaluate a model with a scorer,
# an object called as `scorer(estimator, X, y)`. `make_scorer` turns a metric
# like ours into a scorer. The scorer calls `predict` on `X` and passes only
# the true and predicted labels to the metric. Our metric also needs the
# amount of each test transaction. `Amount` is one of the columns of `X`, but
# the metric never sees `X`.
#
# scikit-learn calls such per-row information *metadata*. Sample weights and
# group labels are other examples. When `cross_validate` splits the data, it
# has to split the metadata the same way, and hand each piece to whichever
# step needs it. This mechanism is called *metadata routing*. We switch it on
# globally, then the scorer declares that it wants the amounts with
# `set_score_request`.

# %%
import sklearn
from sklearn.metrics import make_scorer

sklearn.set_config(enable_metadata_routing=True)
business_scorer = make_scorer(business_gain).set_score_request(amount=True)

# %% [markdown]
# ## Baselines from simple policies
#
# Before training any model, we evaluate policies that need no model at all.
# The first one, "always accept", treats every payment as legitimate. A
# `DummyClassifier` that always predicts 0 implements it. We fit it on the
# first training set and call the scorer ourselves on the first test set,
# passing the amounts of the test rows.

# %%
from sklearn.dummy import DummyClassifier

train, test = next(cv.split(data, target))
always_accept = DummyClassifier(strategy="constant", constant=0)
always_accept.fit(data.iloc[train], target.iloc[train])
gain = business_scorer(
    always_accept, data.iloc[test], target.iloc[test], amount=amount[test]
)
print(f"Gain of 'always accept' on the first test set: {gain:,.0f} €")

# %% [markdown]
# Selecting the right amounts by hand for each split would be tedious.
# `cross_validate` does it for us. We pass it the amounts of all 10,000 rows
# with `params={"amount": amount}`, and for each split it gives the scorer the
# amounts of the test rows only.
#
# We evaluate "always accept" this way, together with "always reject", a
# `DummyClassifier` that always predicts 1. We also compute the gain of an
# oracle that knows which payments are frauds and blocks exactly those. No
# model can reach it, but it tells us how much money is on the table.

# %%
from sklearn.model_selection import cross_validate

test_targets = [target.iloc[test].to_numpy() for _, test in cv.split(data, target)]
test_amounts = [amount[test] for _, test in cv.split(data, target)]

policies = {
    "always accept": DummyClassifier(strategy="constant", constant=0),
    "always reject": DummyClassifier(strategy="constant", constant=1),
}
gains = pd.DataFrame(
    {
        name: cross_validate(
            policy,
            data,
            target,
            cv=cv,
            scoring=business_scorer,
            params={"amount": amount},
        )["test_score"]
        for name, policy in policies.items()
    }
)
gains["oracle"] = [business_gain(y, y, a) for y, a in zip(test_targets, test_amounts)]
gains.agg(["mean", "std"]).round()

# %% [markdown]
# On the first test set, "always accept" loses about 3,000 €, and it loses
# money on every split. This is a consequence of our subsampling. Frauds make
# up 5 percent of our rows, against 0.17 percent in the full dataset, where
# the commissions on legitimate payments outweigh the frauds. All the euros in
# this notebook are therefore per test set of our rebalanced data, and they do
# not estimate what a bank would earn. They remain a fair basis to compare
# models and thresholds with each other.
#
# "Always reject" loses even more. It pays 10 € for each of the 475
# legitimate payments of a test set and never collects any commission. The
# oracle earns about 780 € per test set, namely the commissions on all
# legitimate payments. A useful model has to earn more than "always accept",
# and it cannot earn more than the oracle.

# %% [markdown]
# ## Business gain at the default threshold
#
# We evaluate the same two models as in the previous notebook. Beyond the
# gain at a single threshold, we want to try many thresholds, so we need the
# predicted probabilities of each test set. We fit each model on each split
# and keep only these probabilities. Keeping the fitted models would cost
# memory, because each fitted TabICL model holds its own copy of the
# pretrained weights, more than 100 MB. Each model predicts only once, so we
# also leave the `kv_cache` of the previous notebook off.

# %%
# %pip install -q tabicl
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from tabicl import TabICLClassifier

models = {
    "HGBT": HistGradientBoostingClassifier(random_state=0),
    "TabICL": TabICLClassifier(n_estimators=1, random_state=0),
}
proba = {name: [] for name in models}
for train, test in cv.split(data, target):
    for name, model in models.items():
        fitted = clone(model).fit(data.iloc[train], target.iloc[train])
        proba[name].append(fitted.predict_proba(data.iloc[test])[:, 1])

# %% [markdown]
# `predict` blocks a payment when its predicted probability of fraud is at
# least 0.5. We compute the gain of this default threshold on each test set.

# %%
for name, split_proba in proba.items():
    gains[name] = [
        business_gain(y, p >= 0.5, a)
        for y, a, p in zip(test_targets, test_amounts, split_proba)
    ]
gains.agg(["mean", "std"]).round()

# %% [markdown]
# Both models close about 80 percent of the gap between "always accept" and
# the oracle. On average, gradient boosting earns about 160 € per test set and
# TabICL about 220 €. They still lose money on some splits, three for
# gradient boosting and two for TabICL, because the frauds they miss cost
# more than the commissions they collect. The gap of about 50 € between the
# two models is small compared to the standard deviations across splits,
# about 400 €. We come back to this spread at the end of the notebook.

# %% [markdown]
# ## Business gain across thresholds
#
# The threshold of 0.5 is a convention of `predict`, not a business decision.
# A lower threshold blocks more frauds and more legitimate payments. A higher
# one does the opposite. A ROC curve shows this trade-off in terms of error
# rates. We can draw the same kind of curve with euros on the vertical axis.
#
# We compute the business gain of each test set for a grid of thresholds.
# Both models give many transactions a probability far below 1 percent, so
# we space the thresholds on a log scale.

# %%
import numpy as np

thresholds = np.geomspace(1e-4, 0.9, 80)

gain_curves = {
    name: np.array(
        [
            [business_gain(y, p >= threshold, a) for threshold in thresholds]
            for y, a, p in zip(test_targets, test_amounts, split_proba)
        ]
    )
    for name, split_proba in proba.items()
}

# %% [markdown]
# We draw one panel per model and one thin line per test set, as skore does
# for ROC curves. The legend gives the highest gain of the mean curve, with
# the standard deviation across splits at that threshold. The dotted line marks
# the default threshold of 0.5. We cut the vertical axis at -3,000 €, below
# which TabICL's curves keep falling.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6.75), sharex=True, sharey=True, layout="constrained"
)
for ax, (name, split_gains) in zip(axes, gain_curves.items()):
    best = split_gains.mean(axis=0).argmax()
    for g in split_gains:
        ax.plot(thresholds, g, color="tab:blue", alpha=0.5)
    ax.axvline(0.5, color="black", linestyle=":")
    ax.legend(
        [ax.lines[0], ax.lines[-1]],
        [
            (
                f"best gain={split_gains[:, best].mean():.0f}"
                f"±{split_gains[:, best].std(ddof=1):.0f} € "
                f"at threshold {thresholds[best]:.3f}"
            ),
            "default threshold (0.5)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    ax.set(
        xscale="log",
        ylim=(-3000, None),
        xlabel="decision threshold",
        ylabel="business gain (€)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"estimator = {name}")
fig.suptitle("Business Gain Curve\nPositive label: 1\nData source: Test set")
plt.show()

# %% [markdown]
# The shape of each curve follows from the probabilities we saw on the
# reliability diagrams of the previous notebook.
#
# Gradient boosting's probabilities are nearly binary, with legitimate
# payments close to 0 and frauds close to 1. Only about 7 percent of the
# test transactions get a probability between 0.0001 and 0.5. Moving the
# threshold within this range changes few decisions, so the curves vary
# slowly. On average, they rise from about 160 € at 0.5 to about 260 € at
# 0.001. Over the ten test sets, gradient boosting scores 44 frauds below
# 0.5. Lowering the threshold to 0.001 blocks 18 of them, at the price of 87
# legitimate payments.
#
# TabICL's curves have a plateau between about 2 and 50 percent, highest
# around 3 to 5 percent. They fall steeply below 1 percent. TabICL gives
# legitimate payments small but honest risks. Half of them get a probability
# above 0.15 percent, and one in ten above 0.9 percent. A threshold just
# below 1 percent blocks one legitimate payment in ten, at 10 € each.
#
# The best gains in the legends are optimistic, because we picked the
# threshold on the same test sets we report the gain on. A fair comparison
# has to fix the threshold before looking at the test set. We first recall
# the rule from the previous notebook, which needs no data, then tune a
# threshold on the training rows.

# %% [markdown]
# ## Choosing the threshold without the test set
#
# In the previous notebook, we derived a decision rule from the costs alone.
# A payment of amount `a` is blocked when its probability of fraud exceeds
# `(10 + 0.02 a) / (30 + 1.02 a)`, a break-even risk between 2 and 33
# percent. The rule needs no data, but it reads the probabilities as true
# risks. We compute its gains as a reference for what follows.
#
# ```{note}
# Our probabilities follow the 5 percent fraud rate of our sample, about 30
# times the production rate. Neither this rule nor a threshold tuned on our
# sample would carry over to production as they are.
# ```

# %%
breakeven = [(10 + 0.02 * a) / (30 + 1.02 * a) for a in test_amounts]
gains_from_costs = {
    name: [
        business_gain(y, p >= b, a)
        for y, a, p, b in zip(test_targets, test_amounts, split_proba, breakeven)
    ]
    for name, split_proba in proba.items()
}
pd.DataFrame(gains_from_costs).agg(["mean", "std"]).round()

# %% [markdown]
# As in the previous notebook, the rule earns about 150 € more than the
# default threshold for TabICL, whose plateau contains the break-even risks.
# For gradient boosting, it earns about 50 € more, because many thresholds
# between 0.001 and 0.2 earn about the same, as its gain curves showed.

# %% [markdown]
# The usual alternative is to learn the threshold from the training rows.
# `TunedThresholdClassifierCV` from scikit-learn wraps a classifier and does
# this in three steps:
#
# 1. it runs an internal cross-validation on the training rows, fitting the
#    classifier on part of them and predicting probabilities on the rest;
# 2. for each candidate threshold, it computes a score on these held-out
#    rows and averages it over the internal folds, then keeps the threshold
#    with the best average;
# 3. it refits the classifier on all the training rows, and its `predict`
#    blocks a payment when the probability reaches the chosen threshold.
#
# The score to maximize is the `scoring` parameter. By default it is the
# balanced accuracy, which ignores both the costs and the amounts, so we pass
# our business scorer. Metadata routing does the rest. We pass the amounts of
# the training rows to `fit`, and `TunedThresholdClassifierCV` splits them
# again for its internal folds. Then we call the business scorer on the test
# set, with the amounts of the test rows, as we did for "always accept".
#
# We write the loop over the splits ourselves. `cross_validate` would also
# route the amounts, but to keep the fitted models we would need
# `return_estimator=True`, and ten TabICL models with their own copy of the
# weights would take more than 1 GB. In our loop, only one fitted model
# exists at a time, and we keep what we need from it.
#
# We use the same log-spaced grid of thresholds as above. By default,
# `TunedThresholdClassifierCV` tries 100 thresholds evenly spaced between the
# lowest and highest predicted probabilities, that is, about one every 0.01.
# Such a grid would skip the zone below 1 percent, where the mean gain curve
# of gradient boosting peaks. With `store_cv_results=True`, the tuned model
# keeps the average validation score of each threshold, which we plot below.
#
# The internal cross-validation multiplies the number of fits. To save
# compute, we use 3 internal folds instead of the default 5. Each internal
# validation fold then holds about 500 rows, with about 25 frauds.
#
# ```{tip}
# TabICL takes about a minute to run in this cell on a CPU.
# ```

# %%
from sklearn.model_selection import TunedThresholdClassifierCV

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
tuning = {}
for name, model in models.items():
    rows = []
    for train, test in cv.split(data, target):
        tuned = TunedThresholdClassifierCV(
            model,
            scoring=business_scorer,
            thresholds=thresholds,
            cv=inner_cv,
            store_cv_results=True,
        )
        tuned.fit(data.iloc[train], target.iloc[train], amount=amount[train])
        test_gain = business_scorer(
            tuned, data.iloc[test], target.iloc[test], amount=amount[test]
        )
        rows.append(
            {
                "threshold": tuned.best_threshold_,
                "validation gain": tuned.best_score_,
                "validation curve": tuned.cv_results_["scores"],
                "test gain": test_gain,
            }
        )
    tuning[name] = pd.DataFrame(rows)

# %% [markdown]
# The plot below shows what `TunedThresholdClassifierCV` saw within each of
# the ten training sets. Each line is the gain averaged over the internal
# validation folds, and each dot marks the threshold it picked. An internal
# validation fold holds about 500 rows, as many as a test set, so the gains
# are on the same scale as above. We cut the vertical axis at -1,500 €, below
# which TabICL's curves keep falling.

# %%
fig, axes = plt.subplots(
    1, 2, figsize=(12, 6.75), sharex=True, sharey=True, layout="constrained"
)
for ax, (name, results) in zip(axes, tuning.items()):
    for curve in results["validation curve"]:
        ax.plot(thresholds, curve, color="tab:blue", alpha=0.5)
    ax.plot(
        results["threshold"],
        results["validation gain"],
        "o",
        color="tab:orange",
    )
    ax.axvline(0.5, color="black", linestyle=":")
    ax.legend(
        [ax.lines[0], ax.lines[-2], ax.lines[-1]],
        [
            "mean gain on validation folds",
            "threshold picked in the training set",
            "default threshold (0.5)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    ax.set(
        xscale="log",
        ylim=(-1500, None),
        xlabel="decision threshold",
        ylabel="business gain (€)",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"estimator = {name}")
fig.suptitle("Threshold Tuning\nPositive label: 1\nData source: Training set")
plt.show()

# %% [markdown]
# The validation curves look like the test curves of the same model. The
# shape of each curve comes from the model's probabilities, and it carries
# over from the training rows to new rows.
#
# Gradient boosting's validation curves are nearly flat, so the highest point
# of each curve moves a lot from one training set to the next. TabICL's
# thresholds fall on its plateau, but anywhere along it, from its lower edge
# to beyond 50 percent. We gather the thresholds picked in each split.

# %%
pd.DataFrame({name: results["threshold"] for name, results in tuning.items()}).round(4)

# %% [markdown]
# The thresholds picked for gradient boosting range from 0.02 percent to 57
# percent. Six of them lie between 0.02 and 0.3 percent, far below the
# break-even risks of 2 to 33 percent. TabICL's range from 1.6 to 64
# percent, around and above the break-even range.
#
# We now compare the test gains of the three ways of choosing a threshold.

# %%
threshold_choices = {}
for name in models:
    threshold_choices[(name, "default (0.5)")] = gains[name]
    threshold_choices[(name, "from the costs")] = gains_from_costs[name]
    threshold_choices[(name, "tuned")] = tuning[name]["test gain"]
pd.DataFrame(threshold_choices).agg(["mean", "std"]).T.round()

# %% [markdown]
# Tuning does not pay off here. It earns about as much as the default
# threshold for TabICL, and about 30 € less for gradient boosting. Both tuned
# models earn less than the rule from the costs, by about 80 € for gradient
# boosting and 150 € for TabICL.
#
# The "tuned" row needs care. Each split picked its own threshold, and its
# test set scored that one. The mean is therefore the expected gain of the
# procedure, that is, of running the tuning once and deploying whatever
# threshold it returns. A bank deploys a single threshold, picked on its own
# training data, without knowing in advance which one it will get. To see how
# much this draw matters, we apply each of the ten picked thresholds to all
# ten test sets. We add the default threshold and the rule from the costs,
# which are single choices already.

# %%
single_threshold = {}
for name, results in tuning.items():
    picked = sorted(zip(results["threshold"], results.index))
    single_threshold[name] = pd.DataFrame(
        {
            f"{t:.4f} (split {k})": [
                business_gain(y, p >= t, a)
                for y, a, p in zip(test_targets, test_amounts, proba[name])
            ]
            for t, k in picked
        }
    )
    single_threshold[name]["from the costs"] = gains_from_costs[name]
    single_threshold[name]["default (0.5)"] = gains[name]
pd.concat(
    {name: df.agg(["mean", "std"]).T for name, df in single_threshold.items()}
).round()

# %% [markdown]
# For gradient boosting, the mean gain of a single threshold ranges from
# about 70 € at 0.02 percent to about 270 € at 0.14 percent. For TabICL, it
# ranges from about 160 € at 1.6 percent to about 350 € at 3.2 percent, and
# none beats the rule from the costs. The standard deviations, from about
# 280 € to 440 €, mix two things, namely which frauds fall in each test set,
# a noise that all thresholds share, and how each threshold reacts to them.
#
# To compare the thresholds on an equal footing, we subtract, on each test
# set, the gain of the rule from the costs. The noise shared by all
# thresholds cancels out. A box centred on zero earns as much as the rule,
# and a narrow box earns the same on every test set.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharex=True, layout="constrained")
color = {"whiskers": "black", "medians": "black", "caps": "black"}
for ax, (name, choice_gains) in zip(axes, single_threshold.items()):
    paired = choice_gains.drop(columns="from the costs").sub(
        choice_gains["from the costs"], axis=0
    )
    paired.plot.box(color=color, vert=False, ax=ax)
    ax.axvline(0, color="black", linestyle=":")
    ax.set(xlabel="difference with the rule (€)", title=f"estimator = {name}")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Each threshold against the rule from the costs, on the same test set")
plt.show()

# %% [markdown]
# For gradient boosting, the thresholds above 2 percent, and the default
# one, make almost the same decisions as the rule from the costs, so their
# boxes are narrow and centred on zero. The thresholds below 0.3 percent lose
# between about 40 € and 250 € on a typical test set, because they block
# more legitimate payments. On a few test sets, they win up to about 1,100 €
# by catching a large fraud.
#
# For TabICL, no single threshold beats the rule on a typical test set. The
# thresholds above 10 percent, and the default one, lose up to about 1,300 €
# on one test set, where the rule blocks four frauds that they let through.
# The thresholds between 2 and 6 percent stay within a few hundred euros of
# the rule.
#
# The threshold we want in production earns as much as possible and moves
# little from one test set to the next. For TabICL, the rule from the costs
# is that threshold, and it needs no data. For gradient boosting, the
# thresholds around 0.1 percent earn a little more than the rule on average,
# but we found them by looking at the test sets, which a bank cannot do
# before deploying.
#
# Calibration matters much less once we tune a threshold. A single threshold
# only uses the order of the predictions. If we applied any increasing
# transformation to gradient boosting's probabilities, for example squaring
# them, the tuned threshold would move with them, and every decision would
# stay the same. Tuning the threshold corrects the miscalibration at the one
# point that matters for the decision. What remains is how well each model
# ranks payments, and the previous notebook found both models about equal on
# that front.
#
# Calibration still helps in other ways. First, it gives a threshold without
# any data. For TabICL, the rule from the costs earns more than tuning. The
# same rule also works for gradient boosting, but only because many
# thresholds between 0.001 and 0.2 give it almost the same decisions. Nothing
# in its probabilities told us so in advance.
#
# Second, it removes a risky draw. Tuning on about 25 frauds per validation
# fold can pick a threshold far from the best ones, such as 0.02 percent for
# gradient boosting or 64 percent for TabICL. The rule from the costs never
# depends on such a draw.
#
# Finally, it allows a different threshold for each amount. The rule from the
# costs blocks a small payment only at a higher risk than a large one, which a
# single tuned threshold cannot do. On our data, TabICL's rule earns more
# than any of its single thresholds.

# %% [markdown]
# ## Variability of the gains across splits
#
# The standard deviation across splits is about twice the mean gains. Most
# of this spread comes from frauds that slip through. The simple policies
# already showed it. "Always reject" gives the same result on every test set,
# while the gain of "always accept" varies by about 800 €, because the total
# amount of fraud varies from one test set to the next. Each accepted fraud
# costs its whole amount, while a legitimate payment earns only 2 percent of
# its amount. To win back a fraud of 1,000 €, the bank needs 50,000 € of
# legitimate payments.
#
# We list the frauds above 500 € that at least one model accepts at the
# default threshold, with the probability each model gives them. Splits are
# numbered from 0.

# %%
big_frauds = []
for split, (y, a) in enumerate(zip(test_targets, test_amounts)):
    accepted = (proba["HGBT"][split] < 0.5) | (proba["TabICL"][split] < 0.5)
    is_big = (y == 1) & (a > 500) & accepted
    big_frauds.append(
        pd.DataFrame(
            {
                "split": split,
                "amount": a[is_big],
                **{name: proba[name][split][is_big] for name in models},
            }
        )
    )
pd.concat(big_frauds, ignore_index=True).round(4)

# %% [markdown]
# Both models accept all three. Gradient boosting gives them probabilities
# below 0.3 percent. TabICL gives the fraud of 723 € in split 5 a
# probability of 36 percent, well above its break-even risk of about 3
# percent, so the rule from the costs blocks it. In the same split, the rule
# blocks three smaller frauds that TabICL scores between 5 and 40 percent.
# This is where most of the 150 € that the rule gains for TabICL comes from.
# Gradient boosting scores the same four frauds below 1.3 percent, so the
# rule lets them through.
#
# A fraud that both models accept costs both the same, because both are
# evaluated on the same test sets. Comparing the gains split by split
# removes this shared part of the spread. We compute the difference between
# TabICL and gradient boosting in each split, at the default and at the tuned
# thresholds.

# %%
pd.DataFrame(
    {
        "default threshold": gains["TabICL"] - gains["HGBT"],
        "tuned threshold": tuning["TabICL"]["test gain"] - tuning["HGBT"]["test gain"],
    }
).rename_axis("split").round()

# %% [markdown]
# At the default threshold, the two models make almost the same decisions.
# Their gains differ by less than 35 € on eight test sets out of ten. The two
# exceptions favour TabICL, by about 110 € and 410 €. In split 8, gradient
# boosting lets through eight frauds worth about 950 €, against five worth
# about 610 € for TabICL.
#
# With tuned thresholds, the gaps grow, and they come from the tuning itself.
# In split 1, gradient boosting's threshold of 0.02 percent blocks 28
# legitimate payments, against none for TabICL, and catches only a few small
# frauds in exchange. In split 7, TabICL's threshold of 3.2 percent blocks two
# frauds worth about 450 € that gradient boosting lets through, at the price
# of nine legitimate payments.
#
# With about 25 frauds per test set, a handful of payments decide the
# ranking of the two models, and neither is reliably more profitable at the
# default or at a tuned threshold. The clearest result is elsewhere. With
# TabICL's calibrated probabilities, the rule from the costs beats tuning. To
# be confident about the money, the bank would need many more frauds, or,
# better, a test on live payments.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * wrote a business metric where the gains depend on the amount of each
#   transaction, and passed the amounts to it through `cross_validate` with
#   metadata routing;
# * framed the gains of both models between those of an "always accept"
#   policy and of an oracle;
# * drew the gain of each model across decision thresholds, and related the
#   gentle slope of gradient boosting's curves to its nearly binary
#   probabilities, and the cliff of TabICL's curves to the small but honest
#   risks it gives to legitimate payments;
# * tuned the threshold with `TunedThresholdClassifierCV`, passing the
#   amounts to its internal folds, and saw that a tuned threshold only
#   depends on the ranking of the predictions;
# * found that tuning earns less than the rule from the costs of the
#   previous notebook, which TabICL's calibrated probabilities make
#   possible;
# * applied each tuned threshold to every test set, and compared it with
#   the rule from the costs split by split, to see how much the choice of a
#   single threshold matters;
# * saw that a handful of large frauds drive most of the spread across
#   splits, and compared the two models split by split, which showed that
#   they make almost the same decisions at the default threshold.
