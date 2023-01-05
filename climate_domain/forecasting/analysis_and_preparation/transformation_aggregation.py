from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

file_tag = 'drought'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'

target = 'QV2M'
index = 'date'

data = read_csv(file_path, index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

# sort data by date
data.sort_values(by=data.index.name, inplace=True)


# --------------------- #
# Distribution Original #
# --------------------- #

figure(figsize=(3*HEIGHT, HEIGHT*2))
plot_series(data[target], x_label=index, y_label='consumption', title='Humidity distribution')
plot_series(data['PRECTOT'])
plot_series(data['PS'])
plot_series(data['T2M'])
plot_series(data['T2MDEW'])
plot_series(data['T2MWET'])
plot_series(data['TS'])

xticks(rotation = 45)

savefig(f'images/transformation/original_distribution.png')


# ----------- #
# Aggregation #
# ----------- #

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

# first aggregation
figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, index, 'D')
agg_df.to_csv(f'data/aggregation/{file_tag}_daily_aggregation.csv', index=True)
plot_series(agg_df[target], title='Humidity - Daily measurements', x_label=index, y_label='measurement')
plot_series(agg_df['PRECTOT'], x_label=index, y_label='measurement')
plot_series(agg_df['PS'], x_label=index, y_label='measurement')
plot_series(agg_df['T2M'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MDEW'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MWET'], x_label=index, y_label='measurement')
plot_series(agg_df['TS'], x_label=index, y_label='measurement')

xticks(rotation = 45)
savefig(f'images/transformation/aggregation_daily.png')

# second aggregation
figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, index, 'W')
agg_df.to_csv(f'data/aggregation/{file_tag}_weekly_aggregation.csv', index=True)

plot_series(agg_df[target], title='Humidity - Weekly measurements', x_label=index, y_label='measurement')
plot_series(agg_df[target], title='Humidity - Daily measurements', x_label=index, y_label='measurement')
plot_series(agg_df['PRECTOT'], x_label=index, y_label='measurement')
plot_series(agg_df['PS'], x_label=index, y_label='measurement')
plot_series(agg_df['T2M'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MDEW'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MWET'], x_label=index, y_label='measurement')
plot_series(agg_df['TS'], x_label=index, y_label='measurement')
xticks(rotation = 45)
savefig(f'images/transformation/aggregation_weekly.png')

 # third aggregation
figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, index, 'M')
agg_df.to_csv(f'data/aggregation/{file_tag}_monthly_aggregation.csv', index=True)

plot_series(agg_df[target], title='Humidity - Monthly measurements', x_label=index, y_label='measurement')
plot_series(agg_df['PRECTOT'], x_label=index, y_label='measurement')
plot_series(agg_df['PS'], x_label=index, y_label='measurement')
plot_series(agg_df['T2M'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MDEW'], x_label=index, y_label='measurement')
plot_series(agg_df['T2MWET'], x_label=index, y_label='measurement')
xticks(rotation = 45)
savefig(f'images/transformation/aggregation_monthly.png')
