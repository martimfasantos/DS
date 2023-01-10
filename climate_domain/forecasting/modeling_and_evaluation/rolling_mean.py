from pandas import read_csv, concat, unique, DataFrame
from sklearn.base import RegressorMixin
from ts_functions import split_dataframe, sliding_window
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, split_dataframe
from ds_charts import HEIGHT, multiple_bar_chart
from matplotlib.pyplot import subplots, savefig
from torch import manual_seed, Tensor
from torch.autograd import Variable

file_tag = 'drought'
file_name = f'{file_tag}'
file_path = f'../datasets/{file_name}.csv'

target = 'QV2M'
index_col = 'date'

measure = 'R2'
flag_pct = False


class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame,  Y: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

    
data = read_csv(file_path, index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
# remove non-target columns for profiling
for column in data:
    if column != target:
        data.drop(columns=column, inplace=True)
        
train, test = split_dataframe(data, trn_pct=.70)

sequence_size = [4, 20, 60, 100]

nCols = len(sequence_size)
_, axs = subplots(1, 1, figsize=(HEIGHT, HEIGHT), squeeze=False)
values = {}
best = (0)
last_best = -100
best_model = None
for s in range(len(sequence_size)):
    length = sequence_size[s]
    print(length)
    trnX, trnY = sliding_window(train, seq_length = length)
    trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
    
    tstX, tstY = sliding_window(test, seq_length = length)
    tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))
    
    model = RollingMeanRegressor(win=length)
    model.fit(trnX, trnY)
    prd_trn = model.predict(trnX)
    prd_tst = model.predict(tstX)
    value = (PREDICTION_MEASURES[measure])(tstY, prd_tst)
    
    if value > last_best:
        best = (length)
        last_best = value
        best_model = model
         
    values[length] = value
    #print(values[s])
    
print(sequence_size)
print(values)

multiple_bar_chart(
    sequence_size, values, ax=axs[0, 0], title=f'Rolling mean window size study', xlabel='window size', ylabel=measure, percentage=flag_pct)
print(f'Best results with window size={best} ==> measure={last_best:.2f}')
savefig(f'images/rolling_mean/{file_tag}_rolling_mean_study.png')

# ------------------ #
# Best Results Model #
# ------------------ #

trnX, trnY = sliding_window(train, seq_length = best)
trainY = DataFrame(trnY)
trainY.index = train.index[best+1:]
trainY.columns = [target]
trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
prd_trn = best_model.predict(trnX)
prd_trn = DataFrame(prd_trn)
prd_trn.index=train.index[best+1:]
prd_trn.columns = [target]

tstX, tstY = sliding_window(test, seq_length = best)
testY = DataFrame(tstY)
testY.index = test.index[best+1:]
testY.columns = [target]
tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))
prd_tst = best_model.predict(tstX)
prd_tst = DataFrame(prd_tst)
prd_tst.index=test.index[best+1:]
prd_tst.columns = [target]

plot_evaluation_results(trnY.data.numpy(), prd_trn, tstY.data.numpy(), prd_tst, f'images/rolling_mean/{file_tag}_rolling_mean_eval.png')
plot_forecasting_series(trainY, testY, prd_trn.values, prd_tst.values, f'{file_tag} Rolling Mean Plots', f'images/rolling_mean/{file_tag}_rolling_mean_plots.png', x_label=index_col, y_label=target)
