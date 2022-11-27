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

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins='auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('./images/numeric_granularity_single.png')
# show()


# --------------------------------- #
# Histograms for symbolic variables #
# --------------------------------- #

variables = get_variable_types(data)['Symbolic']
if [] == variables:
    fig, axs = subplots(figsize=(8, 4)) 
    axs.set_title("No Symbolic Variables")          # Do any Matplotlib customization you like
    savefig('./images/granularity_study_symbolic.png')
    # show()
    raise ValueError('There are no symbolic variables.')

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins='auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('./images/granularity_study_symbolic.png')
# show()

# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

variables = get_variable_types(data)['Date']
if [] == variables:
    raise ValueError('There are no date variables.')

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins='auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('./images/granularity_study_date.png')
show()