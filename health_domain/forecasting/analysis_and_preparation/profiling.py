from pandas import Series, read_csv
from numpy import ones
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import subplots
from ds_charts import bar_chart


file_tag = 'glucose'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'

target = 'Glucose'
index_col = 'Date'

data = read_csv(file_path, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

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
print("First timestamp", data.index[0])
print("Last timestamp", data.index[-1])
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='timestamp', y_label='glucose', title='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/dimensionality.png')
# show()

# ---------------- #
# Data Granularity #
# ---------------- #

# Per hours
hour_df = data.copy().groupby([data.index.hour]).mean()
hour_df['timestamp'] = hour_df.index.drop_duplicates()
hour_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(hour_df, hours=True, title='Hourly glucose', x_label='timestamp', y_label='glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_hours.png')
# show()

# Per days
day_df = data.copy().groupby(data.index.date).mean()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(day_df, title='Daily glucose', x_label='timestamp', y_label='glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_days.png')
# show()

# Per weeks
index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly glucose', x_label='timestamp', y_label='glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_weeks.png')
# show()

# Per months
index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Monthly glucose', x_label='timestamp', y_label='glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/profiling/granularity_months.png')
# show()

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
savefig(f'images/profiling/distribution_analysis.png')
# show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
axs[0].title.set_text('Hourly')
data.boxplot(ax=axs[0])
axs[1].title.set_text('Weekly')
week_df.boxplot(ax=axs[1])
savefig(f'images/profiling/distribution.png')
# show()


# ---------------------- #
# Variables Distribution #
# ---------------------- #

bins = ('day', 'week', 'month')
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT*3, 1.5*HEIGHT))

# Per days
day_df = data.copy().groupby(data.index.date)
counts = index.to_series().astype(str).value_counts()
bar_chart(counts.index.to_list(), counts.values, ax=axs[0], title='Histogram for daily Glucose: 149 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[0].tick_params(labelrotation=90)

# Per weeks
index = data.index.to_period('W')
counts = index.to_series().astype(str).value_counts()
bar_chart(counts.index.to_list(), counts.values, ax=axs[1], title='Histogram for weekly Glucose: 22 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[1].tick_params(labelrotation=90)

# Per months
index = data.index.to_period('M')
counts = index.to_series().astype(str).value_counts()
bar_chart(counts.index.to_list(), counts.values, ax=axs[2], title='Histogram for montly Glucose: 5 bins', xlabel='glucose', ylabel='nr records', percentage=False)
axs[2].tick_params(labelrotation=90)

tight_layout()
savefig(f'images/profiling/variable_distribution_granularities.png')


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
plot_series(series, x_label='time', y_label='glucose', title='Stationarity study', show_std=True)
savefig(f'images/profiling/stationarity.png')
# show()
