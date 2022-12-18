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
to_remove = ['fips', 'PRECTOT', 'date', 'Month', 'Year', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ7']

# Gather the variables with meaningful wide distributions
wide_distribution = ['slope6', 'slope7','slope8', 'URB_LAND', 'WAT_LAND', 'CULTIR_LAND']

for el in numeric_vars.copy():
    var = el.split(' ')[0]
    if var in to_remove:
        numeric_vars.remove(el)

# print('Original data:', data.shape)
summary5 = data.describe(include='number')


# ------------------------- #
# APPROACH 1: Drop outliers #
# ------------------------- #

IQR_PARAM = 5
IQR_PARAM_WIDE = 7

df = data.copy(deep=True)

for var in numeric_vars:
    if var.split(' ')[0] in wide_distribution:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM_WIDE)
    else:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
    outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
    # print(f'{var} has {outliers.shape[0]} outliers')
    df.drop(outliers.index, axis=0, inplace=True)
df.to_csv(f'data/outliers/{file_tag}_drop_outliers.csv', index=False)
print('data after dropping outliers:', df.shape)


# ----------------------------- #
# APPROACH 2: Truncate outliers #
# ----------------------------- #

IQR_PARAM = 2
IQR_PARAM_WIDE = 4

df = data.copy(deep=True)

for var in numeric_vars:

    if var.split(' ')[0] in wide_distribution:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM_WIDE)
    else:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

df.to_csv(f'data/outliers/{file_tag}_truncate_outliers.csv', index=False)
# print('data after truncating outliers:', df.describe())
