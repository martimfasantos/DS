from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

register_matplotlib_converters()
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# print(data.shape)

# -------------------------------- #
# Nr of records vs nr of variables #
# -------------------------------- #

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('./images/records_variables.png')
# show()


# -------------- #
# Variables Type #
# -------------- #

variable_types = get_variable_types(data)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('./images/variable_types.png')
# show()


# -------------- #
# Missing values #
# -------------- #

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(5,2))
if (len(mv) == 0):
    fig, axs = plt.subplots(figsize=(8, 4)) 
    axs.set_title("No Missing Values")          # Do any Matplotlib customization you like
else:
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('./images/missing_values.png')
# show()
