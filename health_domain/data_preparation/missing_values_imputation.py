from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, subplots, savefig, show, title
from ds_charts import get_variable_types
from sklearn.impute import SimpleImputer
from numpy import nan
import os


register_matplotlib_converters()
file = 'diabetic_data'
filenames = ['data/variables_encoding/diabetic_data_variables_encoding_1.csv', 'data/variables_encoding/diabetic_data_variables_encoding_1.csv']

# fill with CONSTANT value
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

# fill with MOST FREQUENT value
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


# Strategies for the different variables encoding techniques
for i in (1,2):
    data = read_csv(filenames[i-1], na_values='?', parse_dates=True, infer_datetime_format=True)

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

    # Fill the rest with constant
    df_const = fill_with_constant(df)
    df_const.to_csv(f'data/missing_values/{file}_{i}_drop_columns_then_constant_mv.csv', index=True)

    # Fill the rest with most frequent value
    df_most_freq = fill_with_most_frequent(df)
    df_most_freq.to_csv(f'data/missing_values/{file}_{i}_drop_columns_then_most_frequent_mv.csv', index=True)

    # print(' - Dropping Missing Values - ')
    # print('Dropped variables:', missings)
    # print(f'Original: {data.shape}') 
    # print(f'After: {df.shape}')
    # print('----------------')

    # defines the number of variables to discard entire RECORDS
    threshold = data.shape[1] * 0.50

    # drop records with more missing values than the defined threshold
    df = data.dropna(thresh=threshold, inplace=False)
    # df.to_csv(f'data/missing_values/{file}_{i}_drop_records_mv.csv', index=True)

    ''' NO CHANGES so we do not need to make the fill step since the following
        techinques cover those cases '''

    # print(' - Dropping Missing Values - ')
    # print('Dropped records:', data.shape[0] - df.shape[0])
    # print(f'Original: {data.shape}') 
    # print(f'After: {df.shape}')
    # print('----------------')


    # ---------------------- #
    # Filling Missing Values #
    # ---------------------- #

    # CONSTANT
    df = fill_with_constant(data)
    df.to_csv(f'data/missing_values/{file}_{i}_mv_constant.csv', index=True)
    # df.describe(include='all')


    # MEAN & MOST FREQUENT
    df = fill_with_most_frequent(data)
    df.to_csv(f'data/missing_values/{file}_{i}_mv_most_frequent.csv', index=True)
    # df.describe(include='all')