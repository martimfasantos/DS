from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

file_tag = 'glucose'
file_name = f'{file_tag}_40_smoothing'
file_path = f'data/smoothing/{file_name}.csv'

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


# --------------- #
# Differentiation #
# --------------- #

# first derivative
diff_df = data.diff()
diff_df.to_csv(f'data/differentiation/{file_tag}_1_differentiation.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df['Insulin'], x_label=index, y_label='measurement')
plot_series(diff_df[target], title='Glucose - Differentiation (1st derivative)', x_label=index, y_label='measurement')
xticks(rotation = 45)
tight_layout()
savefig('images/transformation/differentiation_1.png')
# show()

# second derivative
diff_df = diff_df.diff()
diff_df.to_csv(f'data/differentiation/{file_tag}_2_differentiation.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df['Insulin'], x_label=index, y_label='measurement')
plot_series(diff_df[target], title='Glucose - Differentiation (2nd derivative)', x_label=index, y_label='measurement')
xticks(rotation = 45)
tight_layout()
savefig('images/transformation/differentiation_2.png')
