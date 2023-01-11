from numpy import log
from pandas import read_csv, Series
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes, tight_layout
from ds_charts import get_variable_types, choose_grid, bar_chart, multiple_bar_chart, multiple_line_chart, HEIGHT
from seaborn import distplot
import random
from scipy.stats import norm, expon, lognorm
from pandas import to_datetime

register_matplotlib_converters()
filename = '../datasets/drought.csv'
data = read_csv(filename, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

data['date'] = to_datetime(data['date'])

# summary5 = data.describe()
# print(summary5)


# -------------- #
# Global boxplot #
# -------------- #

data.boxplot(rot=45, figsize=(3*HEIGHT, 1.5*HEIGHT))
tight_layout()
savefig('./images/global_boxplot.png')
# show()

# ---------------------------- #
# Boxplots for numeric boxplot #
# ---------------------------- #

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in numeric_vars:
        numeric_vars.remove(v)

if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')
rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0

for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
tight_layout()
savefig('./images/single_boxplots.png')
# show()

#--------- #
# Outliers #
# -------- #

NR_STDEV: int = 2

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in numeric_vars:
        numeric_vars.remove(v)
        
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
        data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(32, HEIGHT)) #figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
tight_layout()
savefig('./images/outliers.png')
# show()

#----------------------- #
# Histograms for numeric #
# ---------------------- #

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in numeric_vars:
        numeric_vars.remove(v)
        
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
tight_layout()
savefig('./images/single_histograms_numeric.png')
# show()


#---------------------------------- #
# Histogram with trend for numeric  #
# --------------------------------- #

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in numeric_vars:
        numeric_vars.remove(v)
        
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
tight_layout()
savefig('./images/histograms_trend_numeric.png')
# show()


#-------------------------- #
# Distributions for numeric #
# ------------------------- #

without_log_norm_dist = ['PRECTOT', 'elevation', 'slope5', 'slope6', 'slope7', 'slope8',
                         'WAT_LAND', 'NVG_LAND', 'URB_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND']
without_exp_dist = []

def compute_known_distributions(x_values: list, log_norm: bool = True, exp: bool = True) -> dict:
    distributions = dict()

    # Gaussian
    mean, sigma = norm.fit(x_values)
    x_values = x_values.tolist()
    random.shuffle(x_values)
    reduced_x_values = x_values[0:999]
    reduced_x_values.sort()
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(reduced_x_values, mean, sigma)

    if exp:
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(reduced_x_values, loc, scale)

    if log_norm:
        #LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(reduced_x_values, sigma, loc, scale)

    return distributions, reduced_x_values


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    if var in without_log_norm_dist and var in without_exp_dist:
        distributions, values = compute_known_distributions(values, False, False)
    elif var in without_log_norm_dist:
        distributions, values = compute_known_distributions(values, False)
    else:
        distributions, values = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

numeric_vars = get_variable_types(data)['Numeric']
binary_vars = get_variable_types(data)['Binary']

for v in binary_vars:
    if v in numeric_vars:
        numeric_vars.remove(v)
        
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
tight_layout()
savefig('./images/histogram_numeric_distribution.png', dpi=90)
# show()


#------------------------ #
# Histograms for symbolic #
# ----------------------- #

symbolic_vars = get_variable_types(data)['Symbolic']
binary_vars = get_variable_types(data)['Binary']

symbolic_vars = symbolic_vars + binary_vars
        
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    bar_chart([str(i) for i in counts.index.to_list()], list(counts.values), ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
tight_layout()
savefig('./images/histograms_symbolic.png')
# show()


#------------------- #
# Class distribution #
# ------------------ #

class_ = data['class'].dropna()
counts = class_.value_counts()
bar_chart(counts.index.to_list(), counts.values, title='Class distribution', xlabel='class', ylabel='nr records', percentage=False)
tight_layout()
savefig('./images/class_distribution.png')
# show()