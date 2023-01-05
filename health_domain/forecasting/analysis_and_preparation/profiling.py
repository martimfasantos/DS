from pandas import Series, read_csv
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import subplots
from numpy import ones

file_tag = 'glucose'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'

target = 'Glucose'

data = read_csv(file_path, index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

# remove non-target columns for profiling
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
# print(data.shape)


# ------------------- #
# Data Dimensionality #
# ------------------- #

print("Nr. Records = ", data.shape[0])
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='timestamp', y_label='quantity', title='Glucose')
xticks(rotation = 45)
tight_layout()
# show()
savefig(f'images/profiling/dimensionality.png')


# ---------------- #
# Data Granularity #
# ---------------- #

# Per days
day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Daily quantities', x_label='timestamp', y_label='quantity')
xticks(rotation = 45)
tight_layout()
# show()
savefig(f'images/profiling/granularity_days.png')

# Per weeks
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly quantities', x_label='timestamp', y_label='quantity')
xticks(rotation = 45)
tight_layout()
# show()
savefig(f'images/profiling/granularity_weeks.png')


# Per months
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Monthly quantities', x_label='timestamp', y_label='quantity')
tight_layout()
# show()
savefig(f'images/profiling/granularity_months.png')

# It does not make sense to do the granularity per quarter since 
# there the data goes from month 03 to month 07.


# ----------------- #
# Data Distribution #
# ----------------- #

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
# show()
savefig(f'images/profiling/distribution_analysis.png')

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
axs[0].title.set_text('Hourly')
data.boxplot(ax=axs[0])
axs[1].title.set_text('Weekly')
week_df.boxplot(ax=axs[1])
# show()
savefig(f'images/profiling/distribution.png')


# ---------------------- #
# Variables Distribution #
# ---------------------- #

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly Glucose %d bins'%bins[j])
    axs[j].set_xlabel('quantity')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data.values, bins=bins[j])
# show()
savefig(f'images/profiling/variable_distribution_hourly.png')

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly Glucose %d bins'%bins[j])
    axs[j].set_xlabel('quantity')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df.values, bins=bins[j])
# show()
savefig(f'images/profiling/variable_distribution_weekly.png')


# ----------------- #
# Data Stationarity #
# ----------------- #

dt_series = Series(data['Glucose'])

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
plot_series(series, x_label='time', y_label='quantities', title='Stationary study', show_std=True)
# show()
savefig(f'images/profiling/stationarity.png')

