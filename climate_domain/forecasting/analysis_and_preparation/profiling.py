from pandas import read_csv, unique
from matplotlib.pyplot import figure, xticks, show, savefig, subplots, tight_layout
from ts_functions import plot_series, HEIGHT
from numpy import ones
from pandas import Series

file_path = f'../datasets/drought_forecasting'

target = 'QV2M'

data = read_csv(f'{file_path}.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

# remove non-target columns for profiling
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
# print(data.shape)


# ------------------- #
# Data Dimensionality #
# ------------------- #

print("Nr. Records = ", data.shape[0])
print("First date", data.index[0])
print("Last date", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='date', y_label='Humidity', title='Drought Dimensionality')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/dimensionality.png')
#show()


# ---------------- #
# Data Granularity #
# ---------------- #

day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Drought Granularity', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity.png')
# show()

# Weekly 
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Drought Granularity', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_weekly.png')
# show()

# Monthly 
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['date'] = index.drop_duplicates().to_timestamp()
month_df.set_index('date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Drought Granularity', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_monthly.png')
# show()

# Quarterly 
index = data.index.to_period('Q')
quarterly_df = data.copy().groupby(index).mean()
quarterly_df['date'] = index.drop_duplicates().to_timestamp()
quarterly_df.set_index('date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(quarterly_df, title='Drought Granularity', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_quarterly.png')
# show()


# ----------------- #
# Data Distribution #
# ----------------- #

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('date', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(3*HEIGHT, HEIGHT/1.9))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
savefig(f'images/profiling/distribution_analysis.png')
# show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
savefig(f'images/profiling/distribution.png')
# show()


# ---------------------- #
# Variables Distribution #
# ---------------------- #

bins = (5, 10, 15)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly meter_reading %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
savefig(f'images/profiling/variables_distribution_hourly.png')
# show()

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly meter_reading %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df.values, bins=bins[j])
savefig(f'images/profiling/variables_distribution_weekly.png')
# show()

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for monthly meter_reading %d bins'%bins[j])
    axs[j].set_xlabel('consumption')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(month_df.values, bins=bins[j])
savefig(f'images/profiling/variables_distribution_monthly.png')
# show()


# ----------------- #
# Data Stationarity #
# ----------------- #

dt_series = Series(data[target])

mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'drought': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='date', y_label='Humidity', title='Stationary study', show_std=True)
savefig(f'images/profiling/stationarity.png')
# show()

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'humidity': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='date', y_label='Humidity', title='Stationary study', show_std=True)
savefig(f'images/profiling/stationarity2.png')
# show()




