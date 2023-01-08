from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import multiple_line_chart
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'drought'
index_col = 'date'
target = 'QV2M'

# Train
train = read_csv(f'../analysis_and_preparation/data/train_and_test/{file_tag}_train.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
train.sort_values(by=train.index.name, inplace=True)

# Test
test = read_csv(f'../analysis_and_preparation/data/train_and_test/{file_tag}_test.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
test.sort_values(by=test.index.name, inplace=True)

# train.index.freq = 'H'
# test.index.freq = 'H'


# ------------------ #
# Random ARIMA Model #
# ------------------ #

pred = ARIMA(train, order=(2, 0, 2))
model = pred.fit(method_kwargs={'warn_convergence': False})
model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))
savefig(f'images/arima/{file_tag}_diagnostics_random.png')


# ----------------- #
# ARIMA Model Study #
# ----------------- #

measure = 'R2'
flag_pct = False
last_best = -100
best = ('',  0, 0.0)
best_model = None

d_values = (0, 1, 2)
params = (1, 2, 3, 5)
ncols = len(d_values)

fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)

for der in range(len(d_values)):
    d = d_values[der]
    values = {}
    for q in params:
        yvalues = []
        for p in params:
            pred = ARIMA(train, order=(p, d, q))
            model = pred.fit(method_kwargs={'warn_convergence': False})
            prd_tst = model.forecast(steps=len(test), signal_only=False)
            yvalues.append(PREDICTION_MEASURES[measure](test,prd_tst))
            if yvalues[-1] > last_best:
                best = (p, d, q)
                last_best = yvalues[-1]
                best_model = model
        values[q] = yvalues
    multiple_line_chart(
        params, values, ax=axs[0, der], title=f'ARIMA d={d}', xlabel='p', ylabel=measure, percentage=flag_pct)
print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')
savefig(f'images/arima/{file_tag}_ts_arima_study.png')
# show()


# ---------------- #
# Best ARIMA Model #
# ---------------- #

prd_trn = best_model.predict(start=0, end=len(train)-1)
prd_tst = best_model.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'images/arima/{file_tag}_arima_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'Drought Best ARIMA Plots', saveto=f'images/arima/{file_tag}_arima_plots.png', x_label= str(index_col), y_label=str(target))