import numpy as np
from pandas import read_csv, concat, unique, DataFrame
from matplotlib.pyplot import figure, savefig, show
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Folder path
dir_path = '../data_preparation/data/outliers/'

# List to store files
file_names = []
file_paths = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
        file_paths.append(f'{dir_path}{file_name}')

target = 'class'
ZERO = 0
ONE = 1

def format_date(date: str):
    if len(date) == len('4022013'):
        day = date[0]
        month = date[1:3]
        year = date[3:]
    else:
        day = date[0:1]
        month = date[2:4]
        year = date[4:]
    return year + "-" + month + "-" + day

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]

    data = read_csv(f'{file_path}.csv')
    unnamed_column = data.columns[0]
    data = data.drop([unnamed_column], axis=1)
        
    values = {'Original': [len(data[data[target] == ZERO]), len(data[data[target] == ONE])]}
    
    train_size = int(0.7 * data.shape[0])
    
    dates = data['date'].copy(deep=True)
    for ind in data.index:
        data['date'][ind] = format_date(str(data['date'][ind]))
    
    # convert to date
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    
    # sort    
    data.sort_values(by='date', inplace=True)
    
    for ind in data.index:
        data['date'][ind] = dates[ind]
    
    train = data.loc[0:train_size-1]
    #print(train)
    
    test = data.loc[train_size:]
    #print(test)
    
    # Train CSV
    train.to_csv(f'data/train_and_test/outliers/{file_name}_train.csv', index=False)

    # Test CSV
    test.to_csv(f'data/train_and_test/outliers/{file_name}_test.csv', index=False)
    
    values['Train'] = [ train[(train['class'] != ZERO)].shape[0], train[(train['class'] != ONE)].shape[0] ]
    values['Test'] = [ test[(test['class'] != ZERO)].shape[0], test[(test['class'] != ONE)].shape[0] ]

    figure(figsize=(12,4))
    multiple_bar_chart([ZERO, ONE], values, title='Data distribution per dataset')
    savefig(f'../data_preparation/images/outliers/distributions_train_test/{file_name}_distribution_train_test.png')
    # # show()