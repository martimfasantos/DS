from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, tight_layout, show
from ds_charts import bar_chart, get_variable_types

register_matplotlib_converters()
filename = '../datasets/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# print(data.shape)

# -------------------------------- #
# Nr of records vs nr of variables #
# -------------------------------- #

figure(figsize=(4,3))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
tight_layout()
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
tight_layout()
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

figure(figsize=(5,4))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
        xlabel='variables', ylabel='nr missing values', rotation=True)
tight_layout()
savefig('./images/missing_values.png')
# show()
