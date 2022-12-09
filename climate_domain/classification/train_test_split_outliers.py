import numpy as np
from pandas import read_csv, concat, unique, DataFrame
from matplotlib.pyplot import figure, savefig, show
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split
import os

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

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]

    data = read_csv(f'{file_path}.csv')

    values = {'Original': [len(data[data[target] == ZERO]), len(data[data[target] == ONE])]}

    y = data.pop(target).values
    X = data.values
    labels = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # Train CSV
    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(f'data/train_and_test/outliers/{file_name}_train.csv', index=False)

    # Test CSV
    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(f'data/train_and_test/outliers/{file_name}_test.csv', index=False)
    
    values['Train'] = [len(np.delete(trnY, np.argwhere((trnY == ZERO)))),
                       len(np.delete(trnY, np.argwhere((trnY == ONE))))]
    values['Test'] = [len(np.delete(tstY, np.argwhere((tstY == ZERO)))),
                      len(np.delete(tstY, np.argwhere((tstY == ONE))))]

    figure(figsize=(12,4))
    multiple_bar_chart([ZERO, ONE], values, title='Data distribution per dataset')
    savefig(f'../data_preparation/images/outliers/distributions_train_test/{file_name}_distribution_train_test.png')
    # show()