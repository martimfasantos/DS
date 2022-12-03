from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, subplots, savefig, show, title
from ds_charts import get_variable_types
from sklearn.impute import SimpleImputer
from numpy import nan

register_matplotlib_converters()
file = 'diabetic_data'
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?', parse_dates=True, infer_datetime_format=True)


# --------------- #
# Missing Values  #
# --------------- #

mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr


# ----------------------- #
# Dropping Missing Values #
# ----------------------- #

# defines the number of records to discard entire COLUMNS
threshold = data.shape[0] * 0.90

# drop columns with more missing values than the defined threshold
missings = [c for c in mv.keys() if mv[c]>threshold]
df = data.drop(columns=missings, inplace=False)
df.to_csv(f'data/{file}_drop_columns_mv.csv', index=True)

# print(' - Dropping Missing Values - ')
# print('Dropped variables:', missings)
# print(f'Original: {data.shape}') 
# print(f'After: {df.shape}')
# print('----------------')

# defines the number of variables to discard entire RECORDS
threshold = data.shape[1] * 0.50

# drop records with more missing values than the defined threshold
df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(f'data/{file}_drop_records_mv.csv', index=True)

# print(' - Dropping Missing Values - ')
# print('Dropped records:', data.shape[0] - df.shape[0])
# print(f'Original: {data.shape}') 
# print(f'After: {df.shape}')
# print('----------------')


# ----------------------- #
# Dropping Missing Values #
# ----------------------- #

# CONSTANT
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
df.to_csv(f'data/{file}_mv_constant.csv', index=True)
# df.describe(include='all')


# MEAN & MOST FREQUENT
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
df.to_csv(f'data/{file}_mv_most_frequent.csv', index=True)
# df.describe(include='all')