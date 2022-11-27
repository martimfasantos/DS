import pandas as pd 
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types



data = pd.read_csv('../datasets/classification/drought.csv', na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = pd.to_datetime(data['date'])


#-----------------------------------Nr Records x Nr Variables-----------------------------------
figure(figsize=(4,4))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('records_variables.png')
show()

#-----------------------------------Nr Variables x Type-----------------------------------

print(data.dtypes)
from pandas import DataFrame
def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64[ns]':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types

variable_types = get_variable_types(data)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('variable_types.png')
show()


#-----------------------------------Missing Values-----------------------------------

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
    savefig('missing_values.png')
    show()
else:
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values')
savefig('missing_values.png')
show()


