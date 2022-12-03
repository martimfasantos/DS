from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ds_charts import get_variable_types
from matplotlib.pyplot import subplots, show, savefig

register_matplotlib_converters()
# file = 'diabetic_data_cleaned'
# filename = '../datasets/classification/diabetic_data_cleaned.csv'
file = 'diabetic_data'
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?', parse_dates=True, infer_datetime_format=True)

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
# TODO: THIS VARIABLES SHOULD BE REMOVED / TURNED DO SYMBOLIC VARS!
numeric_vars.remove('number_outpatient ')
numeric_vars.remove('number_inpatient ')
numeric_vars.remove('number_emergency ')
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']
# No Date variables in this dataset

df_num = data[numeric_vars]
df_symb = data[symbolic_vars]
df_bool = data[boolean_vars]


# --------------------- #
# Z-score Normalization #
# --------------------- #

# scale numeric variables and concat to the rest to create a new csv file
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_symb,  df_bool], axis=1)
norm_data_zscore.to_csv(f'data/{file}_scaled_zscore.csv', index=False)
# print(norm_data_zscore.describe())


# --------------------- #
# MinMax normalization  #
# --------------------- #

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_symb, df_bool], axis=1)
norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
# print(norm_data_minmax.describe())


# ---------- #
# Comparison #
# ---------- #

fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0], rot=45)
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1], rot=45)
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2], rot=45)
savefig(f'images/{file}_scale_comparison.png')
# show()