# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#              OUTLIERS TREATMENT DECISIONS               #
#                                                         #
# Encounter_id: unique identifier of an encounter         #
#   - doesn't make sense to remove (subsequent values)    #
# Patient_nbr: unique identifier of a patient             #
#   - doesn't make sense to remove (subsequent values)    #
# Admission_type_id: [nominal] ex: emergency, elective    #
#   - we don't remove categorical variables               #
# Discharge_disposition_id: [nominal] ex: home, expired   #
#   - we don't remove categorical variables               #
# Admission_source_id: [nominal] ex: physician referral   #
#   - we don't remove categorical variables               #
# Time_in_hospital: days between admission and discharge  #
#   - Remove >= 10 (mean + 2*stdev)
# Num_lab_procedures: lab tests performed                 #
#   - TODO: print counts of the ones > 110                #
# Num_procedures: procedures performed                    #
#   - Remove > 4 (mean + 2*stdev)
# Num_medications: medicines administered                 #
#   - TODO: print counts of the ones > 70                 #
# Number_outpatient: outpatient visits in the prev. year  #
#   - Remove > 30                                         #
# Number_emergency: emercy visits in the prev. year       #
#   - Remove > 30                                         #
# Number_inpatient: inpatient visits in the prev. year    #
#   - TODO: print counts of the ones > 10                 #
# Number_diagnoses: diagnoses entered to the system       #
#   - TODO: print counts of the ones > 11 and < 4         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame
from ds_charts import get_variable_types

# Best option
file_name = 'diabetic_data_drop_columns_then_most_frequent_mv'
file_tag = 'diabetic_data'
file_path = 'data/missing_values/diabetic_data_drop_columns_then_most_frequent_mv.csv'
data = read_csv(file_path, na_values='?')
index_column = data.columns[0]
data = data.drop([index_column], axis=1)

def determine_outlier_thresholds(summary5: DataFrame, var:str, OUTLIER_PARAM: int, option:str):
    if 'iqr' == option:
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%']  + iqr
        bottom_threshold = summary5[var]['25%']  - iqr
    else:
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

# variables in which we'll need to treat outliers
expon_vars = ['encounter_id', 'time_in_hospital', 'number_inpatient', 'race', 'age', 'diag_1', 'diag_2', 'diag_3']
norml_vars = ['num_lab_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_diagnoses']

df = data.copy(deep=True)
summary5 = data.describe(include='number')
print('Original data:', data.shape)


# ------------------------- #
# APPROACH 1: Drop outliers #
# ------------------------- #

IQR_PARAM = 2.5

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in expon_vars + norml_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, IQR_PARAM, 'iqr')
    outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
    df.drop(outliers.index, axis=0, inplace=True)
df.to_csv(f'data/outliers/{file_tag}_drop_outliers.csv', index=True)

print('Data after dropping outliers:', df.shape)


# ----------------------------- #
# APPROACH 2: Truncate outliers #
# ----------------------------- #

IQR_PARAM = 2

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in expon_vars + norml_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, IQR_PARAM, 'iqr')
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)
df.to_csv(f'data/outliers/{file_tag}_truncate_outliers.csv', index=True)

print('Data after truncating outliers:', df.shape)


