from pandas import read_csv, concat, unique, DataFrame
from ts_functions import HEIGHT, PREDICTION_MEASURES
from matplotlib.pyplot import figure, savefig
from ds_charts import bar_chart
from sklearn.base import RegressorMixin

file_tag = 'drought'
file_path = f'../analysis_and_preparation/data/train_and_test/{file_tag}'

target = 'QV2M'
index_col = 'date'

measure = 'R2'
flag_pct = False
eval_results = {}

# Train
train = read_csv(f'{file_path}_train.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
train.sort_values(by=train.index.name, inplace=True)
train.dropna(inplace=True)

# Test
test = read_csv(f'{file_path}_test.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
test.sort_values(by=test.index.name, inplace=True)
test.dropna(inplace=True)


# -------------- #
# Simple Average #
# -------------- #

class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print("SimpleAvg Results: ", eval_results['SimpleAvg'])


# ----------------- #
# Persistence Model #
# ----------------- #

class PersistenceRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print("Persistence Results: ", eval_results['Persistence'])


# ------------- #
# Rolling Mean  #
# ------------- #

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

fr_mod = RollingMeanRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print("RollingMean Results: ", eval_results['RollingMean'])


# ------------------------ #
# Regressors Comparission  #
# ------------------------ #

figure()
bar_chart(list(eval_results.keys()), list(eval_results.values()), title = 'Basic Regressors Comparison', xlabel= 'Regressor', ylabel=measure, percentage=flag_pct, rotation = False)
savefig('images/regressors_comparisson/regressors_comparission.png')