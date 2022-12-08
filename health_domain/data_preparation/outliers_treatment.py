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

file_name = 'diabetic_data_1_drop_columns_then_most_frequent_mv'
file_tag = 'diabetic_data'
file_path = 'data/missing_values/diabetic_data_1_drop_columns_then_most_frequent_mv.csv'
data = read_csv(file_path, na_values='?')
index_column = data.columns[0]
data = data.drop([index_column], axis=1)

from pandas import DataFrame
from ds_charts import get_variable_types

def determine_outlier_thresholds(summary5: DataFrame, var:str,outlier_param: int):
    std = outlier_param * summary5[var]['std']
    top_threshold = summary5[var]['mean'] + std
    bottom_threshold = summary5[var]['mean'] - std
    return top_threshold, bottom_threshold

# variables in which we'll need to treat outliers
numeric_vars = ['time_in_hospital', 'num_lab_procedures', 
                'num_procedures', 'num_medications', 
                'number_outpatient','number_emergency',
                'number_inpatient', 'number_diagnoses']

df = data.copy(deep=True)
summary5 = data.describe(include='number')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# APPROACH 1: dropping outliers beyond 3 stdev          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('Original data:', data.shape)

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 3)
    outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
    df.drop(outliers.index, axis=0, inplace=True)
df.to_csv(f'data/outliers/{file_tag}_drop_outliers.csv', index=True)

print('Data after dropping outliers:', df.shape)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# APPROACH 2: truncating outliers beyond 3 stdev          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

summary5 = data.describe(include='number')
df = data.copy(deep=True)
for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 3)
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)
df.to_csv(f'data/outliers/{file_tag}_truncate_outliers.csv', index=True)

print('Data after truncating outliers:', df.shape)


