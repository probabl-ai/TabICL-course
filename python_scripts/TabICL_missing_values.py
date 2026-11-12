# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # How do HGBT and TabICL cope with missing values?
#
# So far, every table we gave HGBT and TabICL was complete. The learning curve
# took whole rows away, and TabICL's pretraining made up for them. Real tables
# also lose values inside the rows. A weather station goes offline for an
# hour, and the temperature of that hour is never recorded.
#
# Both models accept such holes, written as `NaN`, without asking us to fill
# them first. They handle them in very different ways, though. HGBT learns
# where to send a missing value at each split of its trees. TabICL quietly
# replaces a missing number by the average of its column before its
# transformer reads the table.
#
# In this notebook, we remove a growing share of the weather readings from
# the bike sharing data, from none at all to every one of them, and measure
# how the error of both models grows. We then give TabICL a better guess than
# the average, read from the hour before each hole.

# %% [markdown]
# ## Learning objectives
#
# - Simulate values missing completely at random.
# - Explain how HGBT and TabICL each handle missing values without an
#   imputation step.
# - Measure how the error of both models grows as weather readings go
#   missing.
# - Fill missing values before TabICL inside a pipeline, and measure how much
#   of the lost accuracy a better guess wins back.

# %% [markdown]
# We reuse the bike sharing data and the splitter of the first notebook,
# namely ten folds of 800 training hours and 200 test hours, with a 48-hour
# gap between them.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

bike_sharing = pd.read_csv("../datasets/bike_sharing.csv")
categorical_columns = ["season", "holiday", "workingday", "weather"]
bike_sharing[categorical_columns] = bike_sharing[categorical_columns].astype("category")

target_name = "count"
target = bike_sharing[target_name]
data = bike_sharing.drop(columns=target_name)

ts_cv = TimeSeriesSplit(n_splits=10, gap=48, max_train_size=800, test_size=200)

# %% [markdown]
# ## Removing weather readings completely at random
#
# The calendar columns, such as `hour` or `holiday`, follow from the date, so
# they are never missing in practice. The weather columns come from sensors,
# and sensors fail. We only remove values from `weather`, `temp`,
# `feel_temp`, `humidity`, and `windspeed`, and keep the target complete.
#
# Each value goes missing with the same probability, whatever its value and
# whatever else happens in the row. Such values are **missing completely at
# random** (MCAR). We draw one random number per cell, once, and a cell goes
# missing when its number falls below the chosen fraction. So the cells
# missing at 25 percent are also missing at 50 percent, and the curves below
# compare nested sets of holes rather than unrelated ones. At 100 percent, we
# drop the weather columns, so the models predict from the calendar alone.

# %%
weather_columns = ["weather", "temp", "feel_temp", "humidity", "windspeed"]

rng = np.random.default_rng(0)
random_draws = pd.DataFrame(
    rng.random((len(data), len(weather_columns))),
    columns=weather_columns,
    index=data.index,
)


def remove_weather(data, missing_fraction):
    if missing_fraction == 1:
        return data.drop(columns=weather_columns)
    data_missing = data.copy()
    data_missing[weather_columns] = data[weather_columns].mask(
        random_draws < missing_fraction
    )
    return data_missing


remove_weather(data, 0.5).isna().mean().round(2)

# %% [markdown]
# ## How each model copes with missing readings
#
# HGBT learns where missing values go. At each split, it tries sending the
# rows with a missing value to the left child and to the right child, and
# keeps the side that reduces the error most. For a categorical column such
# as `weather`, a missing value is treated as one more category.
#
# TabICL fills the holes before its transformer reads the table. Its internal
# preprocessing passes the numerical columns through a scikit-learn
# `SimpleImputer`, which replaces each missing value by the mean of the column
# over the training rows. A missing `temp` thus looks like an average
# temperature to the transformer. Categorical columns go through an
# `OrdinalEncoder`, which gives missing values a code of their own.
#
# We fit TabICL many times in this notebook, so we use a single ensemble
# member, `n_estimators=1`, instead of the default 8. Each member costs one
# more pass of the transformer over the rows, as we saw in the exercise on
# `n_estimators`. As a consequence, TabICL's errors on the complete data
# differ slightly from those of the first notebook.
#
# ```{tip}
# This cell takes about 3 minutes on a laptop CPU, almost all of it spent by
# TabICL. On a slow machine, uncomment the shorter list of fractions.
# ```

# %%
# %pip install -q tabicl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from tabicl import TabICLRegressor

missing_fractions = [0, 0.25, 0.5, 0.75, 1]
# missing_fractions = [0, 0.5, 1]


def missingness_curve(models, missing_fractions):
    curve = {}
    for name, model in models.items():
        mae = {}
        for missing_fraction in missing_fractions:
            cv_results = cross_validate(
                model,
                remove_weather(data, missing_fraction),
                target,
                cv=ts_cv,
                scoring="neg_mean_absolute_error",
            )
            mae[missing_fraction] = -cv_results["test_score"]
        curve[name] = pd.DataFrame(mae).T
    return curve


curve = missingness_curve(
    {
        "HGBT": HistGradientBoostingRegressor(random_state=0),
        "TabICL": TabICLRegressor(n_estimators=1, random_state=0),
    },
    missing_fractions,
)

# %% [markdown]
# Each curve is a table with one row per fraction and one column per fold. We
# plot the mean error across folds, with the standard deviation as error bars.

# %%
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
model_colors = {
    "HGBT": "tab:blue",
    "TabICL": "tab:orange",
    "Forward fill + TabICL": "tab:green",
}


def plot_curves(curve):
    _, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for name, mae in curve.items():
        ax.errorbar(
            100 * mae.index,
            mae.mean(axis=1),
            yerr=mae.std(axis=1),
            fmt="o-",
            capsize=3,
            color=model_colors[name],
            label=name,
            alpha=0.7,
        )
    ax.set(
        xlabel="Missing weather values (%)",
        ylabel="Test mean absolute error",
        title="Error as weather values go missing",
        ylim=(0, None),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()


plot_curves(curve)

# %% [markdown]
# Both errors grow as weather values disappear. From complete readings to
# none, HGBT's mean error rises from 42 to 47 bikes per hour, and TabICL's
# from 37 to 45. TabICL keeps the lower error at every fraction, although the
# gap narrows to less than 2 bikes per hour once both models predict from the
# calendar alone.
#
# The two curves bend differently. HGBT's error peaks at 75 percent and is
# slightly lower with no weather at all, so the quarter of readings left
# gives it nothing. TabICL still draws about 2 bikes per hour from that
# quarter, compared with no weather at all.
#
# As in the learning curve, the error bars are wide and overlap. They mostly
# show that some test windows are much harder than others, for both models.

# %% [markdown]
# ## Filling the holes from the previous hour
#
# TabICL's guess for a missing number, the column mean, is the same for every
# hour. Yet the weather changes slowly from one hour to the next, so the last
# reading before a hole is a much better guess. The pandas method `ffill`,
# for forward fill, copies that reading into the hole. It only reads past
# hours, so it uses nothing we would not know at prediction time.
#
# We wrap `ffill` in a `FunctionTransformer` and chain it before TabICL with
# `make_pipeline`. A column transformer restricts it to the numerical
# columns, and passes the categorical ones through unchanged. Its pandas
# output keeps their `category` dtype, and TabICL encodes them, holes
# included, as before. We don't need to scale the filled columns, since
# TabICL standardizes its inputs internally.
#
# The transformer learns nothing in `fit`. It fills the training rows from
# earlier training rows, and the test rows from earlier test rows, which only
# makes sense because both are in time order. A hole in the very first test
# rows has no earlier reading to copy, so it stays empty, and TabICL fills it
# with the mean as before.
#
# At 0 percent there is nothing to fill, and at 100 percent there is nothing
# left to copy from, so we only run the pipeline in between.
#
# ```{tip}
# This cell takes about a minute and a half on a laptop CPU.
# ```

# %%
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def forward_fill(X):
    return X.ffill()


forward_filler = make_column_transformer(
    (FunctionTransformer(forward_fill), make_column_selector(dtype_include="number")),
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

curve |= missingness_curve(
    {
        "Forward fill + TabICL": make_pipeline(
            forward_filler, TabICLRegressor(n_estimators=1, random_state=0)
        )
    },
    missing_fractions[1:-1],
)
plot_curves(curve)

# %% [markdown]
# Forward fill wins back part of what the missing readings cost TabICL, as
# long as the holes stay sparse. At 25 percent, it lowers TabICL's error from
# 39.4 to 37.6 bikes per hour, close to the 37.1 of the complete data. At 50
# percent, it lowers it from 41.9 to 39.4, about half of the way back.
# At 75 percent, holes often span several hours, so the copied reading is
# older, and the gain vanishes.
#
# Both pipelines see the same test windows, so we can also compare them fold
# by fold. We subtract the error of forward fill from the error of TabICL
# alone on each window. A positive gain means that forward fill lowers the
# error.

# %%
forward_fill_mae = curve["Forward fill + TabICL"]
gain = curve["TabICL"].loc[forward_fill_mae.index] - forward_fill_mae
gain.assign(mean=gain.mean(axis=1), std=gain.std(axis=1)).round(1)

# %% [markdown]
# At 25 and 50 percent, the mean gain is about 2 bikes per hour, with a
# standard deviation of about 6. The spread is wide because one test window
# gains 17 to 18 bikes per hour, while the others range from a loss of about
# 7 to a gain of about 6.
#
# Forward fill helps because it brings information that no column of a row
# holds, namely the readings of the previous hours.

# %% [markdown]
# ## Notebook recap
#
# In this notebook, we:
#
# * removed a growing share of the weather values completely at random, with
#   nested sets of holes, up to predicting from the calendar alone;
# * saw that HGBT learns at each split where missing values go, while TabICL
#   replaces a missing number by the column mean through a `SimpleImputer`;
# * measured that both errors grow as readings disappear, with TabICL keeping
#   the lower error at every fraction, and still using the last quarter of
#   the readings where HGBT no longer does;
# * filled the holes before TabICL, inside a pipeline, by copying the last
#   past reading forward, which won back part of the lost accuracy up to 50
#   percent of missing values, but nothing at 75 percent.
