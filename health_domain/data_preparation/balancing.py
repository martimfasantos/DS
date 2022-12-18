from pandas import read_csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from pandas import concat, DataFrame
from pandas import Series
from matplotlib.pyplot import figure, show
from ds_charts import multiple_bar_chart
from imblearn.over_sampling import SMOTE


# Folder path
dir_path = '../data_preparation/data/scaling/'

# List to store files
file_tag = 'diabetic_data'
file_path = f'../classification/data/train_and_test/scaling/{file_tag}_scaled_zscore'

test = read_csv(f'{file_path}_test.csv')
test.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_test.csv', index=False)


# ------------ #
# NO Balancing #
# ------------ #

train = read_csv(f'{file_path}_train.csv')
train.to_csv(f'data/balancing/{file_tag}_without_balancing_train.csv', index=False)
train.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_without_balancing_train.csv', index=False)


# ---------------------- #
# Balancing Training Set #
# ---------------------- #

y = train['readmitted'].values
y_train = pd.Index(y)
target_count = y_train.value_counts().sort_index()
# print(target_count)
labels = ['NO', '<30', '>30']

values = {'Original': [target_count[0], target_count[1], target_count[2]]}
plt.clf()
bar_chart(labels, target_count.values, title='readmitted balance')
savefig(f'images/balancing/{file_tag}_readmitted_balance.png')

readmitted_zero = train[train['readmitted'] == 0]
readmitted_one = train[train['readmitted'] == 1]
readmitted_two = train[train['readmitted'] == 2]


# ----------- #
#    SMOTE    #
# ----------- #

smote = SMOTE(sampling_strategy='all', random_state=8)
y = train.pop('readmitted').values
X = train.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(train.columns) + ['readmitted']
df_smote.to_csv(f'data/balancing/{file_tag}_smote_train.csv', index=False) 
df_smote.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_smote_train.csv', index=False)

smote_target_count = Series(smote_y).value_counts().sort_index()
values['SMOTE'] = [smote_target_count[0], smote_target_count[1], smote_target_count[2]]
print("After Smote:", values['SMOTE'])

figure()
multiple_bar_chart(labels, values, title='SMOTE Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/balancing/{file_tag}_readmitted_smote_balance_bar_chart.png')


# ----------------- #
#    Undersampling  #
# ----------------- #

df_zero_sample = DataFrame(readmitted_zero.sample(len(readmitted_one)))
df_two_sample = DataFrame(readmitted_two.sample(len(readmitted_one)))
df_under = concat([readmitted_one, df_zero_sample, df_two_sample], axis=0)
df_under.to_csv(f'data/balancing/{file_tag}_under_train.csv', index=False)
df_under.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_under_train.csv', index=False)

values['UnderSample'] = [len(df_zero_sample), len(readmitted_one), len(df_two_sample)]
print("After UnderSampling:", values['UnderSample'])
figure()
bar_chart(labels , values['UnderSample'], title='readmitted undersampling balance')
savefig(f'images/balancing/{file_tag}_readmitted_undersampling_balance_bar_chart.png')


# ----------------- #
#    Oversampling   #
# ----------------- #

df_one_sample = DataFrame(readmitted_one.sample(len(readmitted_zero), replace=True))
df_two_sample = DataFrame(readmitted_two.sample(len(readmitted_zero), replace=True))
df_over = concat([ readmitted_zero, df_one_sample, df_two_sample], axis=0)
df_over.to_csv(f'data/balancing/{file_tag}_over_train.csv', index=False)
df_over.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_over_train.csv', index=False)

values['OverSample'] = [len(readmitted_zero), len(df_one_sample), len(df_two_sample)]
print("After OverSampling:", values['OverSample'])
figure()
bar_chart(labels, values['OverSample'], title='readmitted oversampling balance')
savefig(f'images/balancing/{file_tag}_readmitted_oversampling_balance_bar_chart.png')


# -------------------------------- #
#    Oversampling + Undersampling   #
# --------------------------------- #

df_one_sample = DataFrame(readmitted_one.sample(len(readmitted_two), replace=True))
df_zero_sample = DataFrame(readmitted_zero.sample(len(readmitted_two)))
df_mix = concat([df_zero_sample, df_one_sample, readmitted_two], axis=0)
df_mix.to_csv(f'data/balancing/{file_tag}_mix_train.csv', index=False)
df_mix.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_mix_train.csv', index=False)

values['Mix'] = [len(df_zero_sample), len(df_one_sample), len(readmitted_two)]
print("After Mix:", values['Mix'])
figure()
bar_chart(labels, values['Mix'], title='readmitted mix sampling balance')
savefig(f'images/balancing/{file_tag}_readmitted_mixsampling_balance_bar_chart.png')
