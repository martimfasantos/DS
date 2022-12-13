from pandas import DataFrame, read_csv, concat
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from matplotlib.pyplot import figure, savefig, show, subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler

register_matplotlib_converters()

# Choose the BEST MODEL based on results from KNN and NB:
file_tag = 'diabetic_data'
file_path = 'data/outliers/diabetic_data_truncate_outliers.csv'
data = read_csv(file_path, na_values='?')
index_column = data.columns[0]
data = data.drop([index_column], axis=1)

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']
# No Date variables in this dataset

# Since they are unique keys, it does not make sense to consider
# them in the normalization procedure
numeric_vars.remove('patient_nbr')
numeric_vars.remove('encounter_id')

# Variables that do not require normalization for better results
to_remove = ['metformin_variation', 'repaglinide_variation', 'nateglinide_variation', 
             'chlorpropamide_variation', 'glimepiride_variation', 'glipizide_variation', 
             'glyburide_variation', 'pioglitazone_variation', 'rosiglitazone_variation', 
             'acarbose_variation', 'miglitol_variation', 'examide_prescribed', 'examide_variation', 
             'citoglipton_prescribed', 'citoglipton_variation', 'insulin_variation', 
             'glyburide-metformin_variation', 'acetohexamide_variation', 'tolbutamide_variation',
             'troglitazone_variation', 'glipizide-metformin_variation', 'glimepiride-pioglitazone_variation', 
             'metformin-rosiglitazone_variation', 'metformin-pioglitazone_variation'
             'max_glu_serum_level', 'A1Cresult_level', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
             'metformin-pioglitazone_variation', 'max_glu_serum_level']

for el in to_remove:
    if el in numeric_vars:
        numeric_vars.remove(el)

# Remove class (readmitted) from numeric vars
numeric_vars.remove('readmitted')

df_num = data[numeric_vars]
df_symb = data[symbolic_vars]
df_bool = data[boolean_vars]
df_target = data['readmitted']


# --------------------- #
# Z-score Normalization #
# --------------------- #

# scale numeric variables and concat to the rest to create a new csv file
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
temp_norm_data_zscore = concat([tmp, df_symb, df_bool], axis=1)
norm_data_zscore = concat([temp_norm_data_zscore, df_target], axis=1)
norm_data_zscore.to_csv(f'data/scaling/{file_tag}_scaled_zscore.csv', index=False)
# print(norm_data_zscore.describe())


# --------------------- #
# MinMax normalization  #
# --------------------- #

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
temp_norm_data_minmax = concat([tmp, df_symb, df_bool], axis=1)
norm_data_minmax = concat([temp_norm_data_minmax, df_target], axis=1)
norm_data_minmax.to_csv(f'data/scaling/{file_tag}_scaled_minmax.csv', index=False)
# print(norm_data_minmax.describe())


# ---------- #
# Comparison #
# ---------- #

fig, axs = subplots(1, 3, figsize=(50,15),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0], rot=90)
axs[0, 1].set_title('Z-score normalization')
temp_norm_data_zscore.boxplot(ax=axs[0, 1], rot=90)
axs[0, 2].set_title('MinMax normalization')
temp_norm_data_minmax.boxplot(ax=axs[0, 2], rot=90)
savefig(f'images/scaling/{file_tag}_scale_comparison.png')
# show()