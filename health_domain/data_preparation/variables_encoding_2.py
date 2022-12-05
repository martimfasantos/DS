from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import pandas as pd

register_matplotlib_converters()
file = 'diabetic_data'
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# Drop out all records with missing values
data.dropna(inplace=True)

from pandas import DataFrame, concat
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    #dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

variables = get_variable_types(data)
symbolic_vars = variables['Symbolic'] + variables['Binary']

# not all symbolic variables will be dummified

# binary variables: 0 and 1
binary_vars = ('change', 'diabetesMed')
binary_vars_mappings = [[] for _ in range(len(binary_vars))]
for i in range(len(binary_vars)):
    counts = data[binary_vars[i]].value_counts()
    counts = counts.reset_index().values.tolist()
    if (i == 0):
        binary_vars_mappings[i] = [['No', 0], ['Ch', 1]]
    elif (i == 1):
        binary_vars_mappings[i] = [['Yes', 1], ['No', 0]]

X = [0 for _ in range(len(binary_vars))]
for i in range(len(binary_vars)):
    var_mapping = binary_vars_mappings[i]
    X[i] = data[binary_vars[i]]
    for j in var_mapping:
        X[i].replace(to_replace=j[0], value=j[1], inplace=True)
        
other_vars = [c for c in data.columns if not c in binary_vars]
data = concat([data[other_vars], X[0]], axis=1)
data = concat([data, X[1]], axis=1)
    
# ordinal variables: map to an integer which represents the order
ordinal_vars = ('age', 'weight')
ordinal_vars_mappings = [[] for _ in range(len(ordinal_vars))]
for i in range(len(ordinal_vars)):
    counts = data[ordinal_vars[i]].value_counts()
    counts = counts.reset_index().values.tolist()
    counts.sort(key=lambda x: int(x[0].split('-')[0][1:]))
    for j in range(len(counts)):
        counts[j][1] = j
    ordinal_vars_mappings[i] = counts

X = [0 for _ in range(len(ordinal_vars))]
for i in range(len(ordinal_vars)):
    var_mapping = ordinal_vars_mappings[i]
    X[i] = data[ordinal_vars[i]]
    for j in var_mapping:
        X[i].replace(to_replace=j[0], value=j[1], inplace=True)
        
other_vars = [c for c in data.columns if not c in ordinal_vars]
data = concat([data[other_vars], X[0]], axis=1)
data = concat([data, X[1]], axis=1)
    
# variables whose values are in "No", "Steady", "Up", "Down": new mapping
# "Prescribed?", "Variation" (-1,0,+1)
level_vars = ('metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
              'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
              'miglitol', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
              'glyburide-metformin', 'acetohexamide', 'tolbutamide', 'troglitazone',
              'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
              'metformin-pioglitazone')
level_vars_mappings = [["No", [0, 0]], ["Steady", [1, 0]], ["Up", [1, 1]], ["Down", [1, -1]]]

X = [[0, 0] for _ in range(len(level_vars))]
for i in range(len(level_vars)):
    X[i][0] = data[level_vars[i]]
    X[i][0].rename('%s_prescribed' %level_vars[i], inplace=True)
    X[i][1] = data[level_vars[i]].copy(deep=True)
    X[i][1].rename('%s_variation' %level_vars[i], inplace=True)
    
    for j in level_vars_mappings:
        X[i][0].replace(to_replace=j[0], value=j[1][0], inplace=True)
        X[i][1].replace(to_replace=j[0], value=j[1][1], inplace=True)
    
other_vars = [c for c in data.columns if not c in level_vars]
data = concat([data[other_vars], X[0][0]], axis=1)
data = concat([data, X[0][1]], axis=1)
#print(data)
for i in range(1, len(level_vars)):
    data = concat([data, X[i][0]], axis=1)
    data = concat([data, X[i][1]], axis=1)

# variables whose values are in "None", ">x", "Norm", ">y": new mapping
# "Measured?", "Level" (0, +1, +2)
bigger_vars = ('max_glu_serum', 'A1Cresult')
bigger_vars_mappings = [[], []]
bigger_vars_mappings[0] = [["None", [0, -1]], [">200", [1, 2]], [">300", [1, 3]], ["Norm", [1, 1]]]
bigger_vars_mappings[1] = [["None", [0, -1]], [">7", [1, 2]], [">8", [1, 3]], ["Norm", [1, 1]]]

X = [[0, 0] for _ in range(len(bigger_vars))]
for i in range(len(bigger_vars)):
    X[i][0] = data[bigger_vars[i]]
    X[i][0].rename('%s_measured' %bigger_vars[i], inplace=True)
    X[i][1] = data[bigger_vars[i]].copy(deep=True)
    X[i][1].rename('%s_level' %bigger_vars[i], inplace=True)
    
    for j in bigger_vars_mappings[i]:
        X[i][0].replace(to_replace=j[0], value=j[1][0], inplace=True)
        X[i][1].replace(to_replace=j[0], value=j[1][1], inplace=True)
        
other_vars = [c for c in data.columns if not c in bigger_vars]
data = concat([data[other_vars], X[0][0]], axis=1)
data = concat([data, X[0][1]], axis=1)
data = concat([data, X[1][0]], axis=1)
data = concat([data, X[1][1]], axis=1)
    
# class variable
class_var = ('readmitted')
class_var_mapping = [['>30', 0], ['NO', 1], ['<30', 2]]

X = data[class_var]
for j in class_var_mapping:
    X.replace(to_replace=j[0], value=j[1], inplace=True) 
other_vars = [c for c in data.columns if not c in class_var]
data = concat([data[other_vars], X], axis=1)

# dummify the rest
for el in binary_vars:
    symbolic_vars.remove(el)
for el in ordinal_vars:
    symbolic_vars.remove(el)
# for el in our_ordinal_vars:
#     symbolic_vars.remove(el)
for el in bigger_vars:
    symbolic_vars.remove(el)
for el in level_vars:
    symbolic_vars.remove(el)
symbolic_vars.remove('readmitted')

df = dummify(data, symbolic_vars)
df.to_csv(f'data/{file}_variables_encoding_2.csv', index=False)

#df.describe(include=[bool])