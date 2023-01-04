from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT

file_path = f'../datasets/drought_forecasting'

target = 'QV2M'

data = read_csv(f'{file_path}.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(data, x_label='date', y_label='consumption', title='ASHRAE original')
xticks(rotation = 45)
savefig(f'../images/transformation/transformation.png')
#show()


# --------- #
# Smoothing #
# --------- #

WIN_SIZE = 10
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
savefig(f'../images/transformation/smoothing_10.png')
xticks(rotation = 45)
#show()

WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'../images/transformation/smoothing_100.png')
#show()


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
agg_df = aggregate_by(data, 'timestamp', 'D')
plot_series(agg_df, title='Daily consumptions', x_label='date', y_label='Humidity')
xticks(rotation = 45)
savefig(f'../images/transformation/aggregation_daily.png')
#show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'W')
plot_series(agg_df, title='Weekly consumptions', x_label='date', y_label='Humidity')
xticks(rotation = 45)
savefig(f'../images/transformation/aggregation_weekly.png')
#show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'timestamp', 'M')
plot_series(agg_df, title='Monthly consumptions', x_label='date', y_label='Humidity')
xticks(rotation = 45)
savefig(f'../images/transformation/aggregation_monthly.png')
#show()



# --------------- #
# Differentiation #
# --------------- #

diff_df = data.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df, title='Differentiation',  x_label='date', y_label='Humidity')
xticks(rotation = 45)
savefig(f'../images/transformation/differentiation.png')
#show()

