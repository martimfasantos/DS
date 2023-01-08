import sys
import os
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure, savefig, show, subplots
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, split_dataframe
from sklearn.base import RegressorMixin
from ds_charts import bar_chart


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
        if os.path.splitext(file)[0].split('_')[-1] == 'test':
            continue
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

class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]
    
    train = read_csv(f'{file_path}.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    train.sort_values(by=train.index.name, inplace=True)
    train.dropna(inplace=True)

    print(file_name)
    if FLAG == 'differentiation':
        test = read_csv(f'../analysis_and_preparation/data/differentiation/{file_name}_test.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    else:
        test = read_csv('../analysis_and_preparation/data/train_and_test/drought_test.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    test.sort_values(by=test.index.name, inplace=True)
    test.dropna(inplace=True)

    # ----------------- #
    # Persistence Model #
    # ----------------- #
    
    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
    print("Persistence Results: ", eval_results)

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/persistence_model/{FLAG}/{file_name}_persistence_eval.png')
    plot_forecasting_series(train, test, prd_trn, prd_tst, f'{file_name} Persistence Plots', saveto=f'images/persistence_model/{FLAG}/{file_name}_persistence_plots.png', x_label=index_col, y_label=target)


    # -------------- #
    # Simple Average #
    # -------------- #

    fr_mod = SimpleAvgRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
    print("SimpleAvg Results: ", eval_results)

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/simple_average/{FLAG}/{file_name}_simpleAvg_eval.png')
    plot_forecasting_series(train, test, prd_trn, prd_tst, f'{file_name} Simple Average Plots', saveto=f'images/simple_average/{FLAG}/{file_name}_simpleAvg_plots.png', x_label=index_col, y_label=target)


    # ------------- #
    # Rolling Mean  #
    # ------------- #

    fr_mod = RollingMeanRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
    print("RollingMean Results: ", eval_results)

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/rolling_mean/{FLAG}/{file_name}_rollingMean_eval.png')
    plot_forecasting_series(train, test, prd_trn, prd_tst,  f'{file_name} Rolling Mean Plots', saveto=f'images/rolling_mean/{FLAG}/{file_name}_rollingMean_plots.png', x_label=index_col, y_label=target)


    # ----------- #
    # Comparison  #
    # ----------- #

    figure()
    bar_chart(list(eval_results.keys()), list(eval_results.values()), title = 'Basic Regressors Comparison', xlabel= 'Regressor', ylabel=measure, percentage=flag_pct)
    savefig(f'images/{FLAG}_original_distribution.png')


