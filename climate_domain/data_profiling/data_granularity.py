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

variables = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in variables:
        variables.remove(v)
        
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
savefig('./images/granularity_study_numeric.png', dpi=95)
# show()


# --------------------------------- #
# Histograms for symbolic variables #
# --------------------------------- #
from ds_charts import bar_chart

symbolic_vars = get_variable_types(data)['Symbolic']
binary_vars = get_variable_types(data)['Binary']
variables = symbolic_vars + binary_vars
if [] == variables:
    raise ValueError('There are no symbolic variables.')


rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*4, rows*HEIGHT*3), squeeze=False)
i, j = 0, 0
n_graphs = 0
for n in range(len(variables)):
    counts = data[variables[n]].value_counts() # counts for each variable value
    bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' %variables[n], xlabel=variables[n], ylabel='nr records', percentage=False, rotation=45)
    i, j = (i + 1, 0) if (n_graphs + 1) % cols == 0 else (i, j + 1)
    n_graphs += 1  

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