<<<<<<< HEAD
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe

file_tag = 'glucose'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'
=======
from pandas import read_csv, concat, unique, DataFrame
from sklearn.base import RegressorMixin
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'glucose'
file_path = f'../analysis_and_preparation/data/train_and_test/{file_tag}'
>>>>>>> e489b26c67c1d606c7567dbb337d8d13b8171b79

target = 'Glucose'
index_col = 'Date'

<<<<<<< HEAD
data = read_csv(file_path, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# remove non-target columns
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
        
print(data.head())

def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(data, trn_pct=0.75)

def plot_forecasting_series(trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
    _, ax = subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(figname)
    ax.plot(trn.index, trn, label='train', color='b')
    ax.plot(trn.index, prd_trn, '--y', label='train prediction')
    ax.plot(tst.index, tst, label='test', color='g')
    ax.plot(tst.index, prd_tst, '--r', label='test prediction')
    ax.legend(prop={'size': 5})


measure = 'R2'
flag_pct = False
eval_results = {}

from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series
=======
measure = 'R2'
flag_pct = False

>>>>>>> e489b26c67c1d606c7567dbb337d8d13b8171b79

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

<<<<<<< HEAD
=======
    
# Train
train = read_csv(f'{file_path}_train.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
train.sort_values(by=train.index.name, inplace=True)
train.dropna(inplace=True)

# Test
test = read_csv(f'{file_path}_test.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
test.sort_values(by=test.index.name, inplace=True)
test.dropna(inplace=True)


# ------------- #
# Rolling Mean  #
# ------------- #

>>>>>>> e489b26c67c1d606c7567dbb337d8d13b8171b79
fr_mod = RollingMeanRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

<<<<<<< HEAD
eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/rolling_mean/{file_tag}_rollingMean_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'{file_name} Rolling Mean Plots', f'images/rolling_mean/{file_tag}_rollingMean_plots.png', x_label=index_col, y_label=target)
=======
eval_results = PREDICTION_MEASURES[measure](test.values, prd_tst)
print("RollingMean Results: ", eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/rolling_mean/{file_tag}_rollingMean_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst,  f'{file_tag} Rolling Mean Plots', saveto=f'images/rolling_mean/{file_tag}_rollingMean_plots.png', x_label=index_col, y_label=target)
>>>>>>> e489b26c67c1d606c7567dbb337d8d13b8171b79
