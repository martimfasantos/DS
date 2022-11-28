from numpy import log
from pandas import read_csv, Series
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes
from ds_charts import get_variable_types, choose_grid, bar_chart, multiple_bar_chart, multiple_line_chart, HEIGHT
from seaborn import distplot
import random
from scipy.stats import norm, expon, lognorm

register_matplotlib_converters()
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# summary5 = data.describe()
# print(summary5)


# -------------- #
# Global boxplot #
# -------------- #

data.boxplot(rot=45,figsize=(1.5*HEIGHT, 2*HEIGHT))
savefig('./images/global_boxplot.png')
# show()


# ---------------------------- #
# Boxplots for numeric boxplot #
# ---------------------------- #

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


#--------- #
# Outliers #
# -------- #

NR_STDEV: int = 2

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

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
savefig('./images/outliers.png')
# show()


#----------------------- #
# Histograms for numeric #
# ---------------------- #

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s' %numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
savefig('./images/single_histograms_numeric.png')
# show()

#---------------------------------- #
# Histogram with trend for numeric  #
# --------------------------------- #

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


#-------------------------- #
# Distributions for numeric #
# ------------------------- #

# exp_dist = ['admission_type_id', 'discharge_disposition_id', 'num_procedures',
#             'number_outpatient', 'number_emergency', 'number_inpatient']
# log_norm_dist = ['num_lab_procedures', 'num_medications']
# norm_dist = ['number_diagnoses', 'time_in_hospital', 'patient_nbr', 'admission_source_id', 'encounter_id']

# def compute_known_distributions(x_values: list, dist: str) -> dict:
#     distributions = dict()
#     if (dist == 'norm'):
#         # Gaussian
#         mean, sigma = norm.fit(x_values)
#         distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
#     elif (dist == 'lognorm'):
#         # LogNorm
#         sigma, loc, scale = lognorm.fit(x_values)
#         distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
#     elif (dist == 'exp'):
#         # Exponential
#         loc, scale = expon.fit(x_values)
#         distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
#     return distributions

# def histogram_with_distributions(ax: Axes, series: Series, var: str):
#     values = series.sort_values().values
#     ax.hist(values, bins=20, density=True)
#     if var in exp_dist:
#         dist = 'exp'
#     elif var in log_norm_dist:
#         dist = 'lognorm'
#     elif var in norm_dist:
#         dist = 'norm'
#     else:
#         raise ValueError("Unknwon variable name.")
#     distributions = compute_known_distributions(values, dist)
#     multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')


# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# i, j = 0, 0
# for n in range(len(numeric_vars)):
#     histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
#     i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
# savefig('./images/histogram_numeric_distribution2.png')
# # show()
'''
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()

    # Gaussian
    mean, sigma = norm.fit(x_values)
    x_values = x_values.tolist()
    random.shuffle(x_values)
    reduced_x_values = x_values[0:999]
    reduced_x_values.sort()
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(reduced_x_values, mean, sigma)

    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(reduced_x_values, loc, scale)

    #LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(reduced_x_values, sigma, loc, scale)

    return distributions, reduced_x_values


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions, values = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')
    print("Another one...")


numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('./images/histogram_numeric_distribution2.png')
# show()
'''

#------------------------ #
# Histograms for symbolic #
# ----------------------- #

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
savefig('./images/histograms_symbolic.png')
# show()


#------------------- #
# Class distribution #
# ------------------ #

class_ = data['readmitted'].dropna()
rows, cols = choose_grid(1)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

counts = class_.value_counts()
bar_chart(counts.index.to_list(), counts.values, ax=axs[0, 0], title='Class distribution', xlabel='readmitted', ylabel='nr records', percentage=False)

savefig('./images/class_distribution.png')
