from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig, tight_layout
from ts_functions import plot_series, HEIGHT

file_tag = 'glucose'
file_name = f'{file_tag}_10_smoothing'
file_paths = [f'data/smoothing/{file_name}.csv', f'data/train_and_test/{file_tag}_test.csv'] # [train, test]

target = 'Glucose'
index = 'Date'

train = read_csv(file_paths[0], index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
test = read_csv(file_paths[1], index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# sort data by date
train.sort_values(by=train.index.name, inplace=True)
test.sort_values(by=test.index.name, inplace=True)


# --------------- #
# Differentiation #
# --------------- #

# first derivative
diff_df = train.diff()
diff_df.to_csv(f'data/differentiation/{file_tag}_1_differentiation.csv', index=True)
test_df = test.diff()
test_df.to_csv(f'data/differentiation/{file_tag}_1_differentiation_test.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df[target], title='Glucose - Differentiation (1st derivative)', x_label=index, y_label='measurement')
xticks(rotation = 45)
tight_layout()
savefig('images/transformation/differentiation_1.png')
# show()

# second derivative
diff_df = diff_df.diff()
diff_df.to_csv(f'data/differentiation/{file_tag}_2_differentiation.csv', index=True)
test_df = test_df.diff()
test_df.to_csv(f'data/differentiation/{file_tag}_2_differentiation_test.csv', index=True)

figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df[target], title='Glucose - Differentiation (2nd derivative)', x_label=index, y_label='measurement')
xticks(rotation = 45)
tight_layout()
savefig('images/transformation/differentiation_2.png')
