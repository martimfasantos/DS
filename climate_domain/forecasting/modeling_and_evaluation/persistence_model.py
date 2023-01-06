import sys
import os
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure, savefig, show, subplots
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, split_dataframe

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
index_col = 'date'

measure = 'R2'
flag_pct = False
eval_results = {}

class PersistenceRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd


for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]
    
    file_name = file_names[i]
    file_path = file_paths[i]
    
    data = read_csv(f'{file_path}.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    data.sort_values(by=data.index.name, inplace=True)

    
    # remove non-target columns
    for column in data:
        if column != target:
            data.drop(columns=column, inplace=True)

    # TODO: MARTELAR AGGRESSIVO PQ NAO FUNCIONA COM OS PRIMEIROS NAN
    if FLAG == 'differentiation':
        if i == 0:
            data = data[2:] # second derivate has no derivative on the first 2 points
        else: 
            data = data[1:] # first derivate has no derivative on the first point

    train, test = split_dataframe(data, trn_pct=0.75)
    
    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
    print(eval_results)

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/{FLAG}/{file_name}_persistence_eval.png')
    plot_forecasting_series(train, test, prd_trn, prd_tst, f'{file_name} Persistence Plots', saveto=f'images/{FLAG}/{file_name}_persistence_plots.png', x_label=index_col, y_label=target)
