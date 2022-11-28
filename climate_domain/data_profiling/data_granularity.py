from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

register_matplotlib_converters()
filename = '../datasets/classification/drought.csv'
data = read_csv(filename, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'])


# -------------------------------- #
# Histograms for numeric variables #
# -------------------------------- #

numeric_vars = get_variable_types(data)['Numeric']
# TODO is it really numeric??????
binary_vars = get_variable_types(data)['Binary']
variables = numeric_vars + binary_vars
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows = len(variables)
bins = (5, 10, 50)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('./images/granularity_study_numeric.png')
# show()


# --------------------------------- #
# Histograms for symbolic variables #
# --------------------------------- #

variables = get_variable_types(data)['Symbolic']
if [] == variables:
    fig, axs = subplots(figsize=(8, 4)) 
    axs.set_title("No Symbolic Variables")          # Do any Matplotlib customization you like
else:
    rows = len(variables)
    bins = (5, 10, 50)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            print(data[variables[i]].values)
            axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('./images/granularity_study_symbolic.png')
# show()

# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

variables = get_variable_types(data)['Date']
if [] == variables:
    raise ValueError('There are no date variables.')

rows = len(variables)
bins = (20, 60, 100)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
savefig('./images/granularity_study_date.png')
# show()