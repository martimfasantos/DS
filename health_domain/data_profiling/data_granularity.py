from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, choose_grid, HEIGHT
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

# variables = get_variable_types(data)['Symbolic']
# if [] == variables:
#     raise ValueError('There are no numeric variables.')

# rows = len(variables)
# bins = (5, 10, 50)
# cols = len(bins)
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# for i in range(rows):
#     for j in range(cols):
#         axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
#         axs[i, j].set_xlabel(variables[i])
#         axs[i, j].set_ylabel('Nr records')
#         axs[i, j].hist(data[variables[i]].values, bins=bins[j])
# savefig('./images/granularity_study_symbolic.png')
# # show()

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError("There are no symbolic variables.")

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*4, rows*HEIGHT*3), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    if (counts.name in ('age', 'weight')):
        #counts.sort_index()
        counts = counts.reset_index().values.tolist()
        counts.sort(key=lambda x: int(x[0].split('-')[0][1:]))
        x, y = [], []
        for el in counts:
            x.append(el[0])
            y.append(el[1])
        bar_chart(x, y, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
    else:
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('./images/granularity_study_symbolic.png')
# show()


# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

variables = get_variable_types(data)['Date']
if [] == variables:
    fig, axs = subplots(figsize=(8, 4)) 
    axs.set_title("No Date Variables")          # Do any Matplotlib customization you like
else:
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