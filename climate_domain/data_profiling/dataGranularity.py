import pandas as pd 
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
import matplotlib.pyplot as plt

data = pd.read_csv('../datasets/classification/drought.csv', na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = pd.to_datetime(data['date'])

#-----------------------------------Histogram Numeric Variables-----------------------------------
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}


numeric_vars = get_variable_types(data)['Numeric']
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
savefig('numeric_granularity_single.png')
show()

#-----------------------------------Histogram Date Variables-----------------------------------

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
savefig('date_granularity_single.png')
show()

#-----------------------------------Histogram Symbolic Variables-----------------------------------

variables = get_variable_types(data)['Symbolic']
if [] == variables:
    fig, axs = plt.subplots(figsize=(8, 4)) 
    axs.set_title("No Symbolic Variables")          # Do any Matplotlib customization you like
    savefig('symbolic_granularity_single.png')
    show()
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
savefig('symbolic_granularity_single.png')
show()