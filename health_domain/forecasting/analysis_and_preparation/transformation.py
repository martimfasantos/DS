from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

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


# --------------------- #
# Distribution Original #
# --------------------- #

figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data, x_label='Date', y_label='Glucose', title='GLUCOSE original')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/original_distribution.png')
# show()


# ----------- #
# Aggregation #
# ----------- #

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'Date', 'D')
plot_series(agg_df, title='Daily quantities', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/aggregation_daily.png')
# show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'Date', 'W')
plot_series(agg_df, title='Weekly quantities', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/aggregation_weekly.png')
# show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'Date', 'M')
plot_series(agg_df, title='Monthly quantities', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/aggregation_monthly.png')
# show()


# --------- #
# Smoothing #
# --------- #

WIN_SIZE = 10
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/smoothing_10.png')
# show()

WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='Date', y_label='Glucose')
xticks(rotation = 45)
tight_layout()
savefig(f'images/transformation/smoothing_100.png')
# show()


# --------------- #
# Differentiation #
# --------------- #

diff_df = data.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df, title='Differentiation', x_label='Date', y_label='consumption')
xticks(rotation = 45)
savefig(f'images/transformation/differentiation.png')
# show()