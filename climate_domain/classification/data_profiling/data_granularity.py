from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, bar_chart, HEIGHT
from matplotlib.pyplot import subplots, savefig, show, tight_layout

register_matplotlib_converters()
filename = '../datasets/drought.csv'
data = read_csv(filename, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
# data['date'] = to_datetime(data['date'])


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
tight_layout()
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
tight_layout()
savefig('./images/granularity_study_symbolic.png')
# show()

# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

def do_year(counts: list) -> list:
    new_counts = {'2000': 0, '2001': 0, '2002': 0, '2003': 0,
                  '2004': 0, '2005': 0, '2006': 0, '2007': 0,
                  '2008': 0, '2009': 0, '2010': 0, '2011': 0,
                  '2012': 0, '2013': 0, '2014': 0, '2015': 0,
                  '2016': 0 }
    for item in counts:
        year = str(item[0]).split('/')[2]
        try:
            new_counts[year] += item[1]
        except: 
            print(f"ERRO IN DAY: {year}")
    return new_counts

def do_month(counts: list) -> list:
    new_counts = {'01': 0, '02': 0, '03': 0, '04': 0,
                  '05': 0, '06': 0, '07': 0, '08': 0,
                  '09': 0, '10': 0, '11': 0, '12': 0 }
    for item in counts:
        month = str(item[0]).split('/')[1]
        try:
            new_counts[month] += item[1]
        except: 
            print(f"ERRO IN DAY: {month}")
    return new_counts

def do_day(counts: list) -> list:
    new_counts = {'01': 0, '02': 0, '03': 0, '04': 0,
                  '05': 0, '06': 0, '07': 0, '08': 0,
                  '09': 0, '10': 0, '11': 0, '12': 0,
                  '13': 0, '14': 0, '15': 0, '16': 0,
                  '17': 0, '18': 0, '19': 0, '20': 0,
                  '21': 0, '22': 0, '23': 0, '24': 0,
                  '25': 0, '26': 0, '27': 0, '28': 0,
                  '29': 0, '30': 0, '31': 0}
    for item in counts:
        day = str(item[0]).split('/')[0]
        try:
            new_counts[day] += item[1]
        except: 
            print(f"ERRO IN DAY: {day}")
    return new_counts


rows, cols = 2, 2
fig, axs = subplots(rows, cols, figsize=(34, 12), squeeze=False)

# DATE
counts = data['date'].value_counts()
bar_chart(counts.index.to_list(), counts.values, ax=axs[0, 0], title='Histogram for date', xlabel='date', ylabel='nr records', percentage=False, rotation=45)
counts = counts.reset_index().values.tolist()

# YEAR
counts_year = do_year(counts)
x, y = [], []
for key in counts_year:
    x.append(key)
    y.append(counts_year[key])
bar_chart(x, y, ax=axs[0, 1], title='Histogram for year (date) WITH 16 BINS', xlabel='year', ylabel='nr records', percentage=False, rotation=45)

# MONTH
counts_month = do_month(counts)
x, y = [], []
for key in counts_month:
    x.append(key)
    y.append(counts_month[key])
bar_chart(x, y, ax=axs[1, 0], title='Histogram for month WITH 12 BINS', xlabel='month', ylabel='nr records', percentage=False, rotation=45)

# DAY
counts_day = do_day(counts)
x, y = [], []
for key in counts_day:
    x.append(key)
    y.append(counts_day[key])
bar_chart(x, y, ax=axs[1, 1], title='Histogram for day (date) WITH 12 BINS', xlabel='day', ylabel='nr records', percentage=False, rotation=45)

tight_layout()
savefig('./images/granularity_study_date.png')
# show()