from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.impute import SimpleImputer
from numpy import nan


register_matplotlib_converters()
file_tag = 'diabetic_data'
file_path = 'data/variables_encoding/diabetic_data_variables_encoding.csv'

# AUX: Fill with CONSTANT value
def fill_with_constant(data: DataFrame) -> DataFrame:
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    df.index = data.index

    return df

# AUX: Fill with MOST FREQUENT value
def fill_with_most_frequent(data: DataFrame) -> DataFrame:
    tmp_nr, tmp_sb, tmp_bool = None, None, None
    variables = get_variable_types(data)
    numeric_vars = variables['Numeric']
    symbolic_vars = variables['Symbolic']
    binary_vars = variables['Binary']

    tmp_nr, tmp_sb, tmp_bool = None, None, None
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(symbolic_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

    df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
    df.index = data.index

    return df


data = read_csv(file_path, na_values='?', parse_dates=True, infer_datetime_format=True)

# --------------- #
# Missing Values  #
# --------------- #

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr


# ----------------------------------------------------------- #
# APPROACH 1: DROP Missing Values & Fill with CONSTANT Value  #
# ----------------------------------------------------------- #

# defines the number of records to discard entire COLUMNS
threshold = data.shape[0] * 0.90

# drop columns with more missing values than the defined threshold
missings = [c for c in mv.keys() if mv[c]>threshold]
df = data.drop(columns=missings, inplace=False)

# Fill the rest with constant
df_const = fill_with_constant(df)
df_const.to_csv(f'data/missing_values/{file_tag}_drop_columns_then_constant_mv.csv', index=True)


# ----------------------------------------------------------- #
# APPROACH 2: DROP Missing Values & Fill with CONSTANT Value  #
# ----------------------------------------------------------- #

# Fill the rest with most frequent value
df_most_freq = fill_with_most_frequent(df)
df_most_freq.to_csv(f'data/missing_values/{file_tag}_drop_columns_then_most_frequent_mv.csv', index=True)