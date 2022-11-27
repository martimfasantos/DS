from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

register_matplotlib_converters()
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')


# -------------------------------- #
# Histograms for numeric variables #
# -------------------------------- #
variables = get_variable_types(data)['Numeric']
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
    savefig('./images/granularity_study_symbolic.png')
    # show()
    raise ValueError('There are no symbolic variables.')

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
savefig('./images/granularity_study_symbolic.png')
# show()


# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

variables = get_variable_types(data)['Date']
if [] == variables:
    fig, axs = subplots(figsize=(8, 4)) 
    axs.set_title("No Date Variables")          # Do any Matplotlib customization you like
    savefig('./images/granularity_study_date.png')
    # show()
    raise ValueError('There are no date variables.')

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
savefig('./images/granularity_study_date.png')
# show()