from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

register_matplotlib_converters()
filename = '../datasets/classification/drought.csv'
data = read_csv(filename, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
#data['date'] = to_datetime(data['date'])


# -------------------------------- #
# Histograms for numeric variables #
# -------------------------------- #
'''
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
'''
# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

def do_year(counts: list) -> list:
    new_counts = [['2000', 0], ['2001', 0], ['2002', 0], ['2003', 0],
                  ['2004', 0], ['2005', 0], ['2006', 0], ['2007', 0],
                  ['2008', 0], ['2009', 0], ['2010', 0], ['2011', 0],
                  ['2012', 0], ['2013', 0], ['2014', 0], ['2015', 0],
                  ['2016', 0]]
    for item in counts:
        year = str(item[0]).split('/')[2]
        if (year == '2000'):
            new_counts[0][1] += item[1]
        elif (year == '2001'):
            new_counts[1][1] += item[1]
        elif (year == '2002'):
            new_counts[2][1] += item[1]
        elif (year == '2003'):
            new_counts[3][1] += item[1]
        elif (year == '2004'):
            new_counts[4][1] += item[1]
        elif (year == '2005'):
            new_counts[5][1] += item[1]
        elif (year == '2006'):
            new_counts[6][1] += item[1]
        elif (year == '2007'):
            new_counts[7][1] += item[1]
        elif (year == '2008'):
            new_counts[8][1] += item[1]
        elif (year == '2009'):
            new_counts[9][1] += item[1]
        elif (year == '2010'):
            new_counts[10][1] += item[1]
        elif (year == '2011'):
            new_counts[11][1] += item[1]
        elif (year == '2012'):
            new_counts[12][1] += item[1]
        elif (year == '2013'):
            new_counts[13][1] += item[1]
        elif (year == '2014'):
            new_counts[14][1] += item[1]
        elif (year == '2015'):
            new_counts[15][1] += item[1]
        elif (year == '2016'):
            new_counts[16][1] += item[1]
        else:
            print("ERRO IN YEAR")
            print(year)
    return new_counts

def do_month(counts: list) -> list:
    new_counts = [['01', 0], ['02', 0], ['03', 0], ['04', 0],
                  ['05', 0], ['06', 0], ['07', 0], ['08', 0],
                  ['09', 0], ['10', 0], ['11', 0], ['12', 0]]
    for item in counts:
        month = str(item[0]).split('/')[1]
        if (month == '01'):
            new_counts[0][1] += item[1]
        elif (month == '02'):
            new_counts[1][1] += item[1]
        elif (month == '03'):
            new_counts[2][1] += item[1]
        elif (month == '04'):
            new_counts[3][1] += item[1]
        elif (month == '05'):
            new_counts[4][1] += item[1]
        elif (month == '06'):
            new_counts[5][1] += item[1]
        elif (month == '07'):
            new_counts[6][1] += item[1]
        elif (month == '08'):
            new_counts[7][1] += item[1]
        elif (month == '09'):
            new_counts[8][1] += item[1]
        elif (month == '10'):
            new_counts[9][1] += item[1]
        elif (month == '11'):
            new_counts[10][1] += item[1]
        elif (month == '12'):
            new_counts[11][1] += item[1]
        else:
            print("ERRO IN MONTH")
    return new_counts

def do_day(counts: list) -> list:
    new_counts = [['01', 0], ['02', 0], ['03', 0], ['04', 0],
                  ['05', 0], ['06', 0], ['07', 0], ['08', 0],
                  ['09', 0], ['10', 0], ['11', 0], ['12', 0],
                  ['13', 0], ['14', 0], ['15', 0], ['16', 0],
                  ['17', 0], ['18', 0], ['19', 0], ['20', 0],
                  ['21', 0], ['22', 0], ['23', 0], ['24', 0],
                  ['25', 0], ['26', 0], ['27', 0], ['28', 0],
                  ['29', 0], ['30', 0], ['31', 0]]
    for item in counts:
        day = str(item[0]).split('/')[0]
        if (day == '01'):
            new_counts[0][1] += item[1]
        elif (day == '02'):
            new_counts[1][1] += item[1]
        elif (day == '03'):
            new_counts[2][1] += item[1]
        elif (day == '04'):
            new_counts[3][1] += item[1]
        elif (day == '05'):
            new_counts[4][1] += item[1]
        elif (day == '06'):
            new_counts[5][1] += item[1]
        elif (day == '07'):
            new_counts[6][1] += item[1]
        elif (day == '08'):
            new_counts[7][1] += item[1]
        elif (day == '09'):
            new_counts[8][1] += item[1]
        elif (day == '10'):
            new_counts[9][1] += item[1]
        elif (day == '11'):
            new_counts[10][1] += item[1]
        elif (day == '12'):
            new_counts[11][1] += item[1]
        elif (day == '13'):
            new_counts[12][1] += item[1]
        elif (day == '14'):
            new_counts[13][1] += item[1]
        elif (day == '15'):
            new_counts[14][1] += item[1]
        elif (day == '16'):
            new_counts[15][1] += item[1]
        elif (day == '17'):
            new_counts[16][1] += item[1]
        elif (day == '18'):
            new_counts[17][1] += item[1]
        elif (day == '19'):
            new_counts[18][1] += item[1]
        elif (day == '20'):
            new_counts[19][1] += item[1]
        elif (day == '21'):
            new_counts[20][1] += item[1]
        elif (day == '22'):
            new_counts[21][1] += item[1]
        elif (day == '23'):
            new_counts[22][1] += item[1]
        elif (day == '24'):
            new_counts[23][1] += item[1]
        elif (day == '25'):
            new_counts[24][1] += item[1]
        elif (day == '26'):
            new_counts[25][1] += item[1]
        elif (day == '27'):
            new_counts[26][1] += item[1]
        elif (day == '28'):
            new_counts[27][1] += item[1]
        elif (day == '29'):
            new_counts[28][1] += item[1]
        elif (day == '30'):
            new_counts[29][1] += item[1]
        elif (day == '31'):
            new_counts[30][1] += item[1]
        else:
            print("ERRO IN DAY")
    return new_counts

from ds_charts import bar_chart

variables = ['date']

YEAR = 0
MONTH = 1
DAY = 2
granularity = YEAR

rows = len(variables) + 1
cols = 3
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*4, rows*HEIGHT), squeeze=False)

counts = data[variables[0]].value_counts()

bar_chart(counts.index.to_list(), counts.values, ax=axs[0, 0], title='Histogram for %s' %variables[0], xlabel=variables[0], ylabel='nr records', percentage=False, rotation=45)

counts = counts.reset_index().values.tolist()

# year
counts_year = do_year(counts)
x, y = [], []
for el in counts_year:
    x.append(el[0])
    y.append(el[1])
bar_chart(x, y, ax=axs[0, 1], title='Histogram for %s WITH 16 BINS' %variables[0], xlabel=variables[0], ylabel='nr records', percentage=False, rotation=45)

# month
counts_month = do_month(counts)

x, y = [], []
for el in counts_month:
    x.append(el[0])
    y.append(el[1])
bar_chart(x, y, ax=axs[0, 2], title='Histogram for %s WITH 12 BINS' %variables[0], xlabel=variables[0], ylabel='nr records', percentage=False, rotation=45)

# day
counts_day = do_day(counts)

x, y = [], []
for el in counts_day:
    x.append(el[0])
    y.append(el[1])
bar_chart(x, y, ax=axs[1, 0], title='Histogram for %s WITH 12 BINS' %variables[0], xlabel=variables[0], ylabel='nr records', percentage=False, rotation=45)

        
savefig('./images/granularity_study_date.png')
# show()