import sys
import os
import numpy as np
import pandas as pd
from pandas import read_csv, concat, unique, DataFrame
from matplotlib.pyplot import figure, savefig, show
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split

# Parse terminal input
FLAG = ''
valid_flags = ('outliers', 'scaling', 'feature_selection')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python train_test_split.py [outliers|scaling|feature_selection]")
    exit(1)

# Folder path
dir_path = f'../data_preparation/data/{FLAG}/'

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
        
    values = {'Original': [len(data[data[target] == ZERO]), len(data[data[target] == ONE])]}
    
    train_size = round(0.7 * data.shape[0])
    test_size = data.shape[0] - train_size
    
    date_column = ''
    for column in data:
        if column.split(' ')[0] == 'date':
            date_column = column
            break

    if (FLAG == 'outliers'):  
        dates = data[date_column].copy(deep=True)
        for ind in data.index:
            data[date_column][ind] = format_date(str(data[date_column][ind]))
        
        # convert to date
        data[date_column] = pd.to_datetime(data[date_column], format='%Y-%m-%d')
    
    # sort    
    data.sort_values(by=date_column, inplace=True)
    
    if (FLAG == 'outliers'): 
        for ind in data.index:
            data[date_column][ind] = dates[ind]
    
    train = data.head(train_size)
    test = data.tail(test_size)

    # Train CSV
    train.to_csv(f'data/train_and_test/{FLAG}/{file_name}_train.csv', index=False)

    # Test CSV
    test.to_csv(f'data/train_and_test/{FLAG}/{file_name}_test.csv', index=False)
    
    values['Train'] = [ train[(train[target] == ZERO)].shape[0], train[(train[target] == ONE)].shape[0] ]
    values['Test'] = [ test[(test[target] == ZERO)].shape[0], test[(test[target] == ONE)].shape[0] ]

    figure(figsize=(12,4))
    multiple_bar_chart([ZERO, ONE], values, title='Data distribution per dataset')
    savefig(f'../data_preparation/images/{FLAG}/distributions_train_test/{file_name}_distribution_train_test.png')
    # show()