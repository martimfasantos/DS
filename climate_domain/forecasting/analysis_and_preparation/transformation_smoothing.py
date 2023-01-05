from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

file_tag = 'glucose'
file_name = f'{file_tag}_daily_aggregation'
file_path = f'data/aggregation/{file_name}.csv'

target = 'Glucose'
index = 'Date'

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
WIN_SIZE = 10
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
smooth_df.to_csv(f'data/smoothing/{file_tag}_10_smoothing.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df[target], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index, y_label='measurement')
plot_series(smooth_df['Insulin'], x_label=index, y_label='measurement')
xticks(rotation = 45)
savefig(f'images/transformation/smoothing_10.png')

# second window size
WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
smooth_df.to_csv(f'data/smoothing/{file_tag}_100_smoothing.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df[target], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index, y_label='measurement')
plot_series(smooth_df['Insulin'], x_label=index, y_label='measurement')
xticks(rotation = 45)
savefig(f'images/transformation/smoothing_100.png')
