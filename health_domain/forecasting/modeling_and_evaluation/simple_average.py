from pandas import read_csv, concat, unique, DataFrame
from sklearn.base import RegressorMixin
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'glucose'
file_path = f'../analysis_and_preparation/data/train_and_test/{file_tag}'

target = 'Glucose'
index_col = 'Date'

measure = 'R2'
flag_pct = False


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

    
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

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results = PREDICTION_MEASURES[measure](test.values, prd_tst)
print("SimpleAvg Results: ", eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/simple_average/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'{file_tag} Simple Average Plots', saveto=f'images/simple_average/{file_tag}_simpleAvg_plots.png', x_label=index_col, y_label=target)
