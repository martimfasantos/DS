import sys
import os
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure, savefig, show, subplots
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, split_dataframe

file_tag = 'drought'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'

target = 'QV2M'
index = 'date'

data = read_csv(file_path, index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# remove non-target columns for transformation
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
# print(data.shape)

# sort data by date
data.sort_values(by=data.index.name, inplace=True)

train_size = round(0.7 * data.shape[0])
test_size = data.shape[0] - train_size

train = data.head(train_size)
test = data.tail(test_size)

# Train CSV
train.to_csv(f'data/train_and_test/{file_name}_train.csv', index=True)

# Test CSV
test.to_csv(f'data/train_and_test/{file_name}_test.csv', index=True)
