# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # The bike sharing dataset
#
# Several notebooks of this course predict how many bikes are rented each hour
# in Washington D.C. In this notebook, we take a first look at that data and
# save the version that the other notebooks load.
#
# The data comes from Capital Bikeshare, the city's bike-sharing system, and
# is available on OpenML.

# %%
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
).frame

# %%
from skrub import TableReport

TableReport(bike_sharing)

# %% [markdown]
# Each row is one hour between 2011 and 2012.
#
# `season`, `year` (0 for 2011, 1 for 2012), `month`, `hour`, and `weekday`
# place that hour in the calendar. `holiday` and `workingday` flag public
# holidays and working days. `weather`, `temp`, `feel_temp`, `humidity`, and
# `windspeed` describe the conditions outside. `count`, the target of our
# models, is the number of bikes rented that hour. Predicting it lets an
# operator rebalance bikes across stations before demand picks up.
#
# `season`, `holiday`, `workingday`, and `weather` have pandas' `category`
# dtype. The other columns are numbers.

# %%
bike_sharing.dtypes

# %% [markdown]
# ## When people ride
#
# The hour of the day matters most, and how it matters depends on whether it
# is a working day.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
bike_sharing.groupby(["hour", "workingday"], observed=True)["count"].mean().unstack(
    "workingday"
).rename(columns={"True": "working day", "False": "weekend or holiday"}).plot(ax=ax)
ax.set(xlabel="Hour of the day", ylabel="Average bikes rented")
_ = ax.legend(title=None)

# %% [markdown]
# On working days, demand peaks around 8 a.m. and again around 5 to 6 p.m.,
# when people commute. On weekends and holidays, it rises slowly through the
# morning and stays high all afternoon. A model has to learn this interaction
# between `hour` and `workingday` to forecast well.
#
# Demand also grows over the two years, and it is lower in winter.

# %%
weekly_mean = bike_sharing["count"].rolling(24 * 7, center=True).mean()

# Without timestamps, we label the rows that start every third month.
quarter_starts = bike_sharing.groupby(["year", "month"]).head(1).iloc[::3]
tick_labels = [
    f"{2011 + year}-{month:02d}"
    for year, month in quarter_starts[["year", "month"]].to_numpy()
]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(weekly_mean, linewidth=2)
ax.set_xticks(quarter_starts.index, labels=tick_labels)
_ = ax.set(xlabel="Month", ylabel="Bikes rented per hour\n(one-week average)")

# %% [markdown]
# The average hour in 2012 sees about 235 rentals, against 144 in 2011.
# Because of this trend, a model trained on older hours tends to underestimate
# newer ones. That is one reason why the notebooks of this course always test
# on hours that come after the training hours.
#
# Weather matters too. Rain cuts demand roughly in half compared with a clear
# sky. `heavy_rain` occurs in only 3 hours of the whole dataset.

# %%
bike_sharing.groupby("weather", observed=True)["count"].agg(["mean", "size"])

# %% [markdown]
# Finally, the target is skewed. Most hours see fewer than 300 rentals, while
# the busiest one sees 977.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
bike_sharing["count"].plot.hist(bins=50, ax=ax, edgecolor="black")
_ = ax.set(xlabel="Bikes rented in one hour", ylabel="Number of hours")

# %% [markdown]
# ## Missing hours
#
# A handful of hours are missing, so consecutive rows are not always one hour
# apart. The dataset has no timestamp column, but the `hour` column lets us
# find the gaps. Inside an unbroken stretch it goes up by one from row to row,
# wrapping around at midnight, so any other step marks the start of a new
# stretch.

# %%
step = bike_sharing["hour"].diff() % 24
step.value_counts()

# %% [markdown]
# Most steps are one hour. The others split the data into 76 stretches of
# consecutive hours. The longest one runs from April to October 2012.

# %%
stretch_id = (step != 1).cumsum()
stretch_id.value_counts().head()

# %% [markdown]
# The notebooks of this course split the data in time, training on a window of
# hours and testing on the hours that follow, with a 48-hour gap in between.
# We keep every hour up to the end of the longest stretch, which drops the last
# 1,495 hours, from October to December 2012, shaded in the plot below. The
# test windows of the notebooks then all fall inside that stretch, so each one
# spans consecutive hours and the 48-hour gap really is 48 hours.

# %%
end = stretch_id[stretch_id == stretch_id.value_counts().idxmax()].index[-1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(weekly_mean, linewidth=2)
ax.axvspan(end, bike_sharing.index[-1], color="tab:gray", alpha=0.3, label="dropped")
ax.set_xticks(quarter_starts.index, labels=tick_labels)
ax.set(xlabel="Month", ylabel="Bikes rented per hour\n(one-week average)")
_ = ax.legend()

# %%
bike_sharing = bike_sharing.loc[:end]
bike_sharing.shape

# %% [markdown]
# ## Saving the dataset
#
# We save the result as a CSV file in the `datasets` folder.

# %%
bike_sharing.to_csv("../datasets/bike_sharing.csv", index=False)

# %% [markdown]
# A CSV file stores text and doesn't remember pandas dtypes. After loading it,
# the other notebooks therefore convert `season`, `holiday`, `workingday`, and
# `weather` back to the `category` dtype.
