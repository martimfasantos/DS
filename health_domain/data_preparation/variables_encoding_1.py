def has_letter(word: str):
    for char in word:
        if char.isalpha():
            return True
    return False

def mapping_diagnosis(var_name: str, counts: list) -> list:
    new_counts = [['injury', []], ['circulatory', []], ['neoplasms', []], 
                      ['other', []], ['diabetes', []], ['genitourinary', []],
                      ['musculoskeletal', []], ['digestive', []], ['respiratory', []]]
    for item in counts:
        if (has_letter(str(item[0]))):
            new_counts[3][1].append(item[0])
        else:
            code = int(str(item[0][0:3]))
            if   (code >= 800) and (code <= 999):
                new_counts[0][1].append(item[0])
            elif ((code >= 390) and (code <= 459)) or (code == 785):
                new_counts[1][1].append(item[0])
            elif (code >= 140) and (code <= 239):
                new_counts[2][1].append(item[0])
            elif (code == 250):
                new_counts[4][1].append(item[0])
            elif ((code >= 580) and (code <= 629)) or (code == 788):
                new_counts[5][1].append(item[0])
            elif (code >= 710) and (code <= 739):
                new_counts[6][1].append(item[0])
            elif ((code >= 520) and (code <= 579)) or (code == 787):
                new_counts[7][1].append(item[0])
            elif ((code >= 460) and (code <= 519)) or (code == 786):
                new_counts[8][1].append(item[0])
            else:
                new_counts[3][1].append(item[0])
    return new_counts

def mapping(var_name: str, variable) -> list:
    counts = variable.value_counts()
    counts = counts.reset_index().values.tolist()
    if (var_name == 'gender'):
        counts[0][1] = -1
        counts[1][1] = 1
        counts[2][1] = 0
    elif (var_name == 'race'):
        counts[0][1] = 1
        counts[1][1] = 2
        counts[2][1] = 3
        counts[3][1] = 3
        counts[4][1] = 3
    elif (var_name == 'medical_specialty'):
        for c in counts:
            if (c[0] == 'Family/GeneralPractice'):
                c[1] = 1
            elif (c[0] == 'InternalMedicine'):
                c[1] = 2
            elif (c[0][0:7] == 'Surgery'):
                c[1] = 4
            elif (c[0] == 'Cardiology'):
                c[1] = 5
            else:
                c[1] = 3
    elif (var_name == 'diag_1') or (var_name == 'diag_2') or (var_name == 'diag_3'):
        # map codes into diagnoses
        new_counts = mapping_diagnosis(var_name, counts)
        i = 0
        for diag in counts:
            for group in new_counts:
                if (diag[0] in group[1]):
                    counts[i][1] = group[0]
            i += 1
        # map codes into integers (according to diagnosis)
        for diag in counts:
            if (diag[1] == 'injury'):
                diag[1] = 1
            elif (diag[1] == 'circulatory'):
                diag[1] = 2
            elif (diag[1] == 'neoplasms'):
                diag[1] = 3
            elif (diag[1] == 'other'):
                diag[1] = 4
            elif (diag[1] == 'diabetes'):
                diag[1] = 5
            elif (diag[1] == 'genitourinary'):
                diag[1] = 6
            elif (diag[1] == 'musculoskeletal'):
                diag[1] = 7
            elif (diag[1] == 'digestive'):
                diag[1] = 8
            elif (diag[1] == 'respiratory'):
                diag[1] = 9
            else:
                raise ValueError('Unknown diagnosis')
    else:
        raise ValueError('ERROR IN MAPPING')
    return counts

from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
import pandas as pd

register_matplotlib_converters()
file = 'diabetic_data'
filename = '../datasets/classification/diabetic_data.csv'
data = read_csv(filename, na_values='?')

# Drop out all records with missing values
#data.dropna(inplace=True)

data.drop(columns=['payer_code'], inplace=True)

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
    
# variables we made ordinal: level of % of readmissions (according to the paper)
our_ordinal_vars = ('gender', 'race', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3')
our_ordinal_vars_mappings = [[] for _ in range(len(our_ordinal_vars))]
for i in range(len(our_ordinal_vars)):
    our_ordinal_vars_mappings[i] = mapping(our_ordinal_vars[i], data[our_ordinal_vars[i]])

X = [0 for _ in range(len(our_ordinal_vars))]
for i in range(len(our_ordinal_vars)):
    var_mapping = our_ordinal_vars_mappings[i]
    X[i] = data[our_ordinal_vars[i]]
    for j in var_mapping:
        X[i].replace(to_replace=j[0], value=j[1], inplace=True)
        
other_vars = [c for c in data.columns if not c in our_ordinal_vars]
data = concat([data[other_vars], X[0]], axis=1)
for i in range(1, len(X)):
    data = concat([data, X[i]], axis=1)

# variables whose values are in "No", "Steady", "Up", "Down": new mapping
# "Prescribed?", "Variation" (-1,0,+1)
level_vars = ('metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
              'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
              'miglitol', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
              'glyburide-metformin', 'acetohexamide', 'tolbutamide', 'troglitazone',
              'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
              'metformin-pioglitazone')
level_vars_mappings = [["No", [0, -2]], ["Steady", [1, 0]], ["Up", [1, 1]], ["Down", [1, -1]]]

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
for el in our_ordinal_vars:
    symbolic_vars.remove(el)
for el in bigger_vars:
    symbolic_vars.remove(el)
for el in level_vars:
    symbolic_vars.remove(el)
symbolic_vars.remove('readmitted')

df = dummify(data, symbolic_vars)
df.to_csv(f'data/{file}_variables_encoding_1.csv', index=False)

#df.describe(include=[bool])