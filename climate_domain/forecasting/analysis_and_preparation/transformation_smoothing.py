from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

file_tag = 'drought'
file_name = f'{file_tag}_daily_aggregation'
file_path = f'data/aggregation/{file_name}.csv'

target = 'QV2M'
index = 'date'

data = read_csv(file_path, index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# remove non-target columns for transformation
# for column in data:
#     if column != target:
#         data.drop(columns=column, inplace=True)
# print(data.shape)

# sort data by date
data.sort_values(by=data.index.name, inplace=True)


# --------- #
# Smoothing #
# --------- #

# first window size
WIN_SIZES = (10, 30, 100, 200)

for win_size in WIN_SIZES:
    rolling = data.rolling(window=win_size, min_periods=1)
    smooth_df = rolling.mean()
    smooth_df.to_csv(f'data/smoothing/{file_tag}_{win_size}_smoothing.csv', index=True)

    figure(figsize=(3*HEIGHT, HEIGHT/2))
    # plot_series(smooth_df['PRECTOT'], x_label=index, y_label='measurement')
    # plot_series(smooth_df['PS'], x_label=index, y_label='measurement')
    # plot_series(smooth_df['T2M'], x_label=index, y_label='measurement')
    # plot_series(smooth_df['T2MDEW'], x_label=index, y_label='measurement')
    # plot_series(smooth_df['T2MWET'], x_label=index, y_label='measurement')
    # plot_series(smooth_df['TS'], x_label=index, y_label='measurement')
    plot_series(smooth_df[target], title=f'Humidity - Smoothing (win_size={win_size})', x_label=index, y_label='measurement')
    xticks(rotation = 45)
    tight_layout()
    savefig(f'images/transformation/smoothing_{win_size}.png')
