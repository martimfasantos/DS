from pandas import DataFrame, read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types

register_matplotlib_converters()
file_tag = 'drought'
file_path = 'data/variables_encoding/drought_variables_encoding.csv'
data = read_csv(file_path, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
index_column = data.columns[0]
data = data.drop([index_column], axis = 1)
# print(data.describe())


# variables that have a Normal distribution
norm_dist_variables = ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
                       'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE' ]


def determine_outlier_thresholds(summary5: DataFrame, var: str, OPTION: str, OUTLIER_PARAM: int):
    # default parameter
    if OPTION == 'iqr':
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%'] + iqr
        bottom_threshold = summary5[var]['25%'] - iqr
    # for normal distribution
    elif OPTION == 'stdev':
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    else:
        raise ValueError('Unknown outlier parameter!')
    return top_threshold, bottom_threshold

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

# Remove non numeric variables (ordinal but not numeric)
to_remove = ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ7']

for el in to_remove:
    if el in numeric_vars:
        numeric_vars.remove(el)

print('Original data:', data.shape)
summary5 = data.describe(include='number')


# ------------------------- #
# APPROACH 1: Drop outliers #
# ------------------------- #

STDEV_PARAM = 3
IQR_PARAM = 10

df = data.copy(deep=True)

for var in numeric_vars:
    if var in norm_dist_variables:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'stdev', STDEV_PARAM)
    else:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
    outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
    df.drop(outliers.index, axis=0, inplace=True)
df.to_csv(f'data/outliers/{file_tag}_drop_outliers.csv', index=True)
print('data after dropping outliers:', df.shape)


# ----------------------------- #
# APPROACH 2: Truncate outliers #
# ----------------------------- #

IQR_PARAM = 8

df = data.copy(deep=True)

for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', 8)
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

df.to_csv(f'data/outliers/{file_tag}_truncate_outliers.csv', index=True)
# print('data after truncating outliers:', df.describe())
