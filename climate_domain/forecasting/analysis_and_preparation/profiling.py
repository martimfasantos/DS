from pandas import read_csv, unique
from matplotlib.pyplot import figure, xticks, show, savefig, subplots, tight_layout
from ts_functions import plot_series, HEIGHT
from numpy import ones
from pandas import Series
from ds_charts import bar_chart

file_path = f'../datasets/drought'

target = 'QV2M'

data = read_csv(f'{file_path}.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# remove non-target columns for profiling
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
# print(data.shape)

# sort data by date
data.sort_values(by=data.index.name, inplace=True)

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

# Daily 
#day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, title='Daily drought', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_daily.png')
# show()

# Weekly 
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['date'] = index.drop_duplicates().to_timestamp()
week_df.set_index('date', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly drought', x_label='date', y_label='Humidity')
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
plot_series(month_df, title='Monthly drought', x_label='date', y_label='Humidity')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_monthly.png')
# show()

# Quarterly 
# index = data.index.to_period('Q')
# quarterly_df = data.copy().groupby(index).mean()
# quarterly_df['date'] = index.drop_duplicates().to_timestamp()
# quarterly_df.set_index('date', drop=True, inplace=True)
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(quarterly_df, title='Drought Granularity', x_label='date', y_label='Humidity')
# xticks(rotation = 45)
# tight_layout()
# savefig(f'images/profiling/granularity_quarterly.png')
# show()


# ----------------- #
# Data Distribution #
# ----------------- #

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)

index = data.index.to_period('M')
month_df = data.copy().groupby(index).sum()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('DAILY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))

axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))

axs[2].grid(False)
axs[2].set_axis_off()
axs[2].set_title('MONTHLY', fontweight="bold")
axs[2].text(0, 0, str(month_df.describe()))

tight_layout()
savefig(f'images/profiling/distribution_analysis.png')

# show()

# Boxplot for the most atomic granularity
_, axs = subplots(1, 1, figsize=(2*HEIGHT, HEIGHT))
axs.title.set_text('DAILY')
data.boxplot(ax=axs)
tight_layout()
savefig(f'images/profiling/distribution.png')
# show()

# ---------------------- #
# Variables Distribution #
# ---------------------- #

bins = ('day', 'week', 'month')
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT*6, 3*HEIGHT))

# Per days
index = data.index.to_period('D')
counts = index.to_series().astype(str).value_counts()
print(counts)
bar_chart(counts.index.to_list(), counts.values, ax=axs[0], title='Histogram for daily Glucose: 7671 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[0].tick_params(labelrotation=90)

# Per weeks
index = data.index.to_period('W')
counts = index.to_series().astype(str).value_counts()
print(counts)
bar_chart(counts.index.to_list(), counts.values, ax=axs[1], title='Histogram for weekly Glucose: 1097 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[1].tick_params(labelrotation=90)

# Per months
index = data.index.to_period('M')
counts = index.to_series().astype(str).value_counts()
print(counts)
bar_chart(counts.index.to_list(), counts.values, ax=axs[2], title='Histogram for monthly Glucose: 252 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[2].tick_params(labelrotation=90)

tight_layout()
savefig(f'images/profiling/variable_distribution_granularities.png')

# ----------------- #
# Data Stationarity #
# ----------------- #

dt_series = Series(data[target])

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'values': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='date', y_label='Humidity', title='Stationary study', show_std=True)
savefig(f'images/profiling/stationarity.png')
# show()




