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
        if (file_name == 'diabetic_data_scaled_zscore'):

            file_names.append(file_name)
            file_paths.append(f'../classification/data/train_and_test/scaling/{file_name}')


for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]

    # ---------------------- #
    # Balancing Training Set #
    # ---------------------- #

    train = read_csv(f'{file_path}_train.csv')

    y = train['readmitted'].values
    y_train = pd.Index(y)
    target_count = y_train.value_counts()  

    print('Readmitted 1 =', target_count.values[0])
    print('Readmitted 0 = ', target_count.values[1])
    print('Readmitted 2 = ', target_count.values[2])
    values = {'Original': [target_count[1],target_count[0], target_count[2]]}
    plt.clf()
    bar_chart([1, 0, 2], target_count.values, title='readmitted balance')
    savefig(f'images/balancing/{file_name}_readmitted_balance.png')

    readmitted_one = train[train['readmitted'] == 1]
    readmitted_zero = train[train['readmitted'] == 0]
    readmitted_two = train[train['readmitted'] == 2]



    # ----------- #
    #    SMOTE    #
    # ----------- #


    smote = SMOTE(sampling_strategy='all', random_state=RANDOM_STATE)
    y = train.pop('readmitted').values
    X = train.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(train.columns) + ['readmitted']
    df_smote.to_csv(f'data/balancing/{file_name}_smote_train.csv', index=False) 

    print("after SMOTE:\n")
    print(Series(smote_y).value_counts())
    smote_target_count = Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[0], smote_target_count[1], smote_target_count[2]]

    figure()
    multiple_bar_chart([1,  0,  2], values, title='SMOTE Target', xlabel='frequency', ylabel='Class balance')
    savefig(f'images/balancing/{file_name}_readmitted_smote_balance_bar_chart.png')


    # ----------------- #
    #    Undersampling  #
    # ----------------- #


    df_one_sample = DataFrame(readmitted_one.sample(len(readmitted_two)))
    df_zero_sample = DataFrame(readmitted_zero.sample(len(readmitted_two)))
    df_under = concat([readmitted_two, df_one_sample, df_zero_sample], axis=0)
    df_under.to_csv(f'data/balancing/{file_name}_under_train.csv', index=False)
    values['UnderSample'] = [len(df_zero_sample), len(df_one_sample), len(readmitted_two)]
   

    print("After UnderSampling :", values['UnderSample'])
    figure()
    bar_chart([1, 0, 2] , values['UnderSample'], title='readmitted undersampling balance')
    savefig(f'images/balancing/{file_name}_readmitted_undersampling_balance_bar_chart.png')


    # ----------------- #
    #    Oversampling   #
    # ----------------- #


    df_two_sample = DataFrame(readmitted_two.sample(len(readmitted_one), replace=True))
    df_zero_sample = DataFrame(readmitted_zero.sample(len(readmitted_one), replace=True))
    df_over = concat([df_zero_sample, df_one_sample, readmitted_two], axis=0)
    df_over.to_csv(f'data/balancing/{file_name}_over_train.csv', index=False)
    values['OverSample'] = [len(df_zero_sample), len(readmitted_one), len(df_two_sample)]
   

    print("After OverSampling :", values['OverSample'])
    figure()
    bar_chart([1, 0, 2] , values['OverSample'], title='readmitted oversampling balance')
    savefig(f'images/balancing/{file_name}_readmitted_oversampling_balance_bar_chart.png')


    # -------------------------------- #
    #    Oversampling + Undersampling   #
    # --------------------------------- #


    df_two_sample = DataFrame(readmitted_two.sample(len(readmitted_zero), replace=True))
    df_one_sample = DataFrame(readmitted_one.sample(len(readmitted_zero)))
    df_over = concat([readmitted_zero, df_one_sample, df_two_sample], axis=0)
    df_over.to_csv(f'data/balancing/{file_name}_mix_train.csv', index=False)
    values['Mix'] = [len(readmitted_zero), len(df_one_sample), len(df_two_sample)]
   

    print("After Mix :", values['Mix'])
    figure()
    bar_chart([1, 0, 2] , values['Mix'], title='readmitted mix sampling balance')
    savefig(f'images/balancing/{file_name}_readmitted_mixsampling_balance_bar_chart.png')





