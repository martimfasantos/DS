from pandas import read_csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from sklearn.model_selection import train_test_split
from pandas import concat, DataFrame
from pandas import Series
from matplotlib.pyplot import figure, show
from ds_charts import multiple_bar_chart
from imblearn.over_sampling import SMOTE
RANDOM_STATE = 42

# Folder path
dir_path = '../data_preparation/data/scaling/'

# List to store files
file_tag = 'drought'
file_path = f'../classification/data/train_and_test/scaling/{file_tag}_scaled_zscore'

test = read_csv(f'{file_path}_test.csv')
test.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_test.csv', index=False)

target = 'class'

# ---------------------- #
# Balancing Training Set #
# ---------------------- #

train = read_csv(f'{file_path}_train.csv')

y = train['class'].values
y_train = pd.Index(y)
target_count = y_train.value_counts()  

print('class 1 =', target_count.values[1])
print('class 0 = ', target_count.values[0])
values = {'Original': [target_count[0], target_count[1]]}
plt.clf()
bar_chart(['0', '1'], target_count.values, title='class balance')
savefig(f'images/balancing/{file_tag}_class_balance.png')

class_one = train[train['class'] == 1]
class_zero = train[train['class'] == 0]

# ----------- #
#    SMOTE    #
# ----------- #

smote = SMOTE(sampling_strategy='all', random_state=RANDOM_STATE)
y = train.pop('class').values
X = train.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(train.columns) + ['class']
df_smote.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_smote_train.csv', index=False)

print("after SMOTE:\n")
print(Series(smote_y).value_counts())
smote_target_count = Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count[0], smote_target_count[1]]

figure()
multiple_bar_chart(['0',  '1'], values, title='SMOTE Target', xlabel='frequency', ylabel='Class balance')
savefig(f'images/balancing/{file_tag}_class_smote_balance_bar_chart.png')

# ----------------- #
#    Undersampling  #
# ----------------- #

df_zero_sample = DataFrame(class_zero.sample(len(class_one)))
df_under = concat([df_zero_sample, class_one], axis=0)
df_under.to_csv(f'data/balancing/{file_tag}_under_train.csv', index=False)
df_under.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_under_train.csv', index=False)

values['UnderSample'] = [len(df_zero_sample), len(class_one)]
print("After UnderSampling :", values['UnderSample'])
figure()
bar_chart(['0', '1'] , values['UnderSample'], title='class undersampling balance')
savefig(f'images/balancing/{file_tag}_class_undersampling_balance_bar_chart.png')


# ----------------- #
#    Oversampling   #
# ----------------- #

df_one_sample = DataFrame(class_one.sample(len(class_zero), replace=True))
df_over = concat([class_zero, df_one_sample], axis=0)
df_over.to_csv(f'data/balancing/{file_tag}_over_train.csv', index=False)
df_over.to_csv(f'../classification/data/train_and_test/balancing/{file_tag}_over_train.csv', index=False)

values['OverSample'] = [len(class_zero), len(df_one_sample)]
print("After OverSampling :", values['OverSample'])
figure()
bar_chart(['0', '1'] , values['OverSample'], title='class oversampling balance')
savefig(f'images/balancing/{file_tag}_class_oversampling_balance_bar_chart.png')





