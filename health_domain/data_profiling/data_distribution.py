from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show
from ds_charts import HEIGHT

data.boxplot(rot=45,figsize=(HEIGHT, 2*HEIGHT))
savefig('images/global_boxplot.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError("There are no numeric variables.")
rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s' %numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('images/single_boxplots.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import figure, savefig, show
from ds_charts import get_variable_types, multiple_bar_chart, HEIGHT

NR_STDEV: int = 2

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError("There are no numeric variables.")

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%'] + iqr].count()[var] +
        data[data[var] < summary5[var]['25%'] - iqr].count()[var]
    ]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]
    ]
    
outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(20, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
savefig('images/outliers.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import get_variable_types, choose_grid, HEIGHT

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError("There are no numeric variables.")

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s' %numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('images/single_histograms_numeric.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show, subplots
from seaborn import distplot
from ds_charts import HEIGHT, get_variable_types

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError("There are no numeric variables.")

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s' %numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('images/histograms_trend_numeric.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm
from matplotlib.pyplot import savefig, show, subplots, Axes
from ds_charts import HEIGHT, multiple_line_chart, get_variable_types

def compute_known_distributions(x_values: list, dist: str) -> dict:
    distributions = dict()
    if (dist == 'norm') or (dist == 'lognorm'):
        # Gaussian
        mean, sigma = norm.fit(x_values)
        distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    elif (dist == 'lognorm'):
        # LogNorm
        sigma, loc, scale = lognorm.fit(x_values)
        distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    elif (dist == 'exp'):
        # Exponential
        loc, scale = expon.fit(x_values)
        distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, bins=20, density=True)
    if (var == 'admission_type_id') or (var == 'discharge_disposition_id') \
        or (var == 'num_procedures') or (var == 'number_outpatient') \
        or (var == 'number_emergency') or (var == 'number_inpatient'):
        dist = 'exp'
    elif (var == 'num_lab_procedures') or (var == 'num_medications'):
        dist = 'lognorm'
    elif (var == 'number_diagnoses') or (var == 'time_in_hospital') \
        or (var == 'patient_nbr') or (var == 'admission_source_id') \
        or (var == 'encounter_id'):
        dist = 'norm'
    else:
        raise ValueError("Unknwon variable name.")
    distributions = compute_known_distributions(values, dist)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/histogram_numeric_distribution.png')
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, choose_grid, get_variable_types, bar_chart

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError("There are no symbolic variables.")

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*2, rows*HEIGHT*2), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    if ('?' in counts.index):
        counts = counts.drop('?')
    bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('images/histograms_symbolic.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import choose_grid, HEIGHT

class_ = data['readmitted']
rows, cols = choose_grid(1)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

counts = class_.value_counts()
if ('?' in counts.index):
        counts = counts.drop('?')
bar_chart(counts.index.to_list(), counts.values, ax=axs[0, 0], title='Class distribution', xlabel='readmitted', ylabel='nr records', percentage=False)

savefig('images/class_distribution.png')
