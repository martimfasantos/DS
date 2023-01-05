import sys
import os
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure, savefig, show, subplots
from ts_functions import HEIGHT

# Parse terminal input
FLAG = ''
valid_flags = ('differentiation', 'smoothing', 'aggregation')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python train_test_split.py [aggregation|smoothing|differentiation]")
    exit(1)

# Folder path
dir_path = f'../analysis_and_preparation/data/{FLAG}/'

# List to store files
file_names = []
file_paths = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
        file_paths.append(f'{dir_path}{file_name}')

target = 'QV2M'
index = 'date'

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]
    
    file_name = file_names[i]
    file_path = file_paths[i]
    data = read_csv(f'{file_path}.csv', index_col=index, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    data.sort_values(by=data.index.name, inplace=True)
    
    # remove non-target columns
    for column in data:
        if column != target:
            data.drop(columns=column, inplace=True)

    # Create lagged dataset
    values = DataFrame(data.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    print(dataframe.head(5))
    
    # split into train and test sets
    X = dataframe.values
    train_size = int(len(X) * 0.7)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]
    
    # persistence model
    def model_persistence(x):
        return x
    
    # walk-forward validation
    predictions = list()
    for x in test_X:
        yhat = model_persistence(x)
        predictions.append(yhat)
    test_score = mean_squared_error(test_y, predictions)
    print('Test RMSE: %.3f' % test_score)
    
    _, axs = subplots(1, 1, figsize=(HEIGHT, HEIGHT/2))
    axs.grid(False)
    axs.set_axis_off()
    axs.set_title('Test MSE', fontweight="bold")
    axs.text(0, 0, 'Test MSE: %.3f' % test_score)
    # show()
    savefig(f'images/{FLAG}/{file_name}_persistence_modelrmse.png')

    # plot predictions and expected results
    figure()
    plt.plot(train_y)
    plt.plot([None for i in train_y] + [x for x in test_y], label="Train")
    plt.plot([None for i in train_y] + [x for x in predictions], label="Test")
    plt.title("Time Series Forecasting")
    plt.legend()
    savefig(f'images/{FLAG}/{file_name}_persistence_model.png')
