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
file_names = []
file_paths = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_name = os.path.splitext(file)[0]
        print(file_name)
        if (file_name == 'drought_scaled_zscore'):

            file_names.append(file_name)
            file_paths.append(f'../classification/data/train_and_test/scaling/{file_name}')


target = 'class'

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]

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
    bar_chart([0, 1], target_count.values, title='class balance')
    savefig(f'images/balancing/{file_name}_class_balance.png')

    class_one = train[train['class'] == 1]
    class_zero = train[train['class'] == 0]


'''
    # ----------- #
    #    SMOTE    #
    # ----------- #


    smote = SMOTE(sampling_strategy='all', random_state=RANDOM_STATE)
    y = train.pop('class').values
    X = train.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(train.columns) + ['class']
    df_smote.to_csv(f'data/balancing/{file_name}_smote_train.csv', index=False) 

    print("after SMOTE:\n")
    print(Series(smote_y).value_counts())
    smote_target_count = Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[0], smote_target_count[1], smote_target_count[2]]

    figure()
    multiple_bar_chart([1,  0,  2], values, title='SMOTE Target', xlabel='frequency', ylabel='Class balance')
    savefig(f'images/balancing/{file_name}_class_smote_balance_bar_chart.png')


    # ----------------- #
    #    Undersampling  #
    # ----------------- #


    df_one_sample = DataFrame(class_one.sample(len(class_two)))
    df_zero_sample = DataFrame(class_zero.sample(len(class_two)))
    df_under = concat([class_two, df_one_sample, df_zero_sample], axis=0)
    df_under.to_csv(f'data/balancing/{file_name}_under_train.csv', index=False)
    values['UnderSample'] = [len(df_zero_sample), len(df_one_sample), len(class_two)]
   

    print("After UnderSampling :", values['UnderSample'])
    figure()
    bar_chart([1, 0, 2] , values['UnderSample'], title='class undersampling balance')
    savefig(f'images/balancing/{file_name}_class_undersampling_balance_bar_chart.png')


    # ----------------- #
    #    Oversampling   #
    # ----------------- #


    df_two_sample = DataFrame(class_two.sample(len(class_one), replace=True))
    df_zero_sample = DataFrame(class_zero.sample(len(class_one), replace=True))
    df_over = concat([df_zero_sample, df_one_sample, class_two], axis=0)
    df_over.to_csv(f'data/balancing/{file_name}_over_train.csv', index=False)
    values['OverSample'] = [len(df_zero_sample), len(class_one), len(df_two_sample)]
   

    print("After OverSampling :", values['OverSample'])
    figure()
    bar_chart([1, 0, 2] , values['OverSample'], title='class oversampling balance')
    savefig(f'images/balancing/{file_name}_class_oversampling_balance_bar_chart.png')


    # -------------------------------- #
    #    Oversampling + Undersampling   #
    # --------------------------------- #


    df_two_sample = DataFrame(class_two.sample(len(class_zero), replace=True))
    df_one_sample = DataFrame(class_one.sample(len(class_zero)))
    df_over = concat([class_zero, df_one_sample, df_two_sample], axis=0)
    df_over.to_csv(f'data/balancing/{file_name}_mix_train.csv', index=False)
    values['Mix'] = [len(class_zero), len(df_one_sample), len(df_two_sample)]
   

    print("After Mix :", values['Mix'])
    figure()
    bar_chart([1, 0, 2] , values['Mix'], title='class mix sampling balance')
    savefig(f'images/balancing/{file_name}_class_mixsampling_balance_bar_chart.png')
'''




