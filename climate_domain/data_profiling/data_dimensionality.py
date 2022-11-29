from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

register_matplotlib_converters()
filename = '../datasets/classification/drought.csv'
data = read_csv(filename, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'])

# print(data.shape)

# -------------------------------- #
# Nr of records vs nr of variables #
# -------------------------------- #

figure(figsize=(4,3))
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
    #bar_chart([], [], title='Nr of missing values per variable',
           # xlabel='no missing values', ylabel='nr missing values')
    fig, axs = plt.subplots(figsize=(8, 4)) 
    axs.set_title("No Missing Values")          # Do any Matplotlib customization you like
else:
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values')
savefig('./images/missing_values.png')
# show()
