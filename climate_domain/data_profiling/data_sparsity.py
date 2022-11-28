from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show, title, figure
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap

register_matplotlib_converters()
data = read_csv('../datasets/classification/drought.csv', na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'])


# ----------------------------------- #
# Scatter all x all - including class #
# ----------------------------------- #

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']
symbolic_vars = get_variable_types(data)['Symbolic']
date_vars = get_variable_types(data)['Date']
all_vars = numeric_vars + binary_vars + date_vars + symbolic_vars
# print(all_vars)
if [] == all_vars:
    raise ValueError('There are no variables.')

all_vars1 = all_vars[:len(all_vars)//3]
rows, cols = len(all_vars1)-1, len(all_vars1)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(all_vars1)):
    var1 = all_vars1[i]
    for j in range(i+1, len(all_vars1)):
        var2 = all_vars1[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig('./images/sparsity_study1.png')
# show()

all_vars2 = all_vars[len(all_vars)//3:int(len(all_vars)//(3/2))]
rows, cols = len(all_vars2)-1, len(all_vars2)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(all_vars2)):
    var1 = all_vars2[i]
    for j in range(i+1, len(all_vars2)):
        var2 = all_vars2[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig('./images/sparsity_study2.png')
# show()

all_vars3 = all_vars[int(len(all_vars)//(3/2)):]
rows, cols = len(all_vars3)-1, len(all_vars3)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(all_vars3)):
    var1 = all_vars3[i]
    for j in range(i+1, len(all_vars3)):
        var2 = all_vars3[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig('./images/sparsity_study3.png')
# show()


# ------------------- #
# Correlation HeatMap #
# ------------------- #

corr_mtx = abs(data.corr())
# print(corr_mtx)

fig = figure(figsize=[50, 50])
heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig('./images/correlation_analysis.png')
# show()
