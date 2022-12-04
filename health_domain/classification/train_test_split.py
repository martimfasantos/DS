import numpy as np
from pandas import read_csv, concat, unique, DataFrame
from matplotlib.pyplot import figure, savefig, show
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split
import os

# Folder path
dir_path = '../data_preparation/data/missing_values/'

# List to store files
file_tags = []
filenames = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_tag = os.path.splitext(file)[0]
        file_tags.append(file_tag)
        filenames.append(f'{dir_path}{file_tag}')

target = 'readmitted'
NO = 1
LESSTHAN30 = 2
MORETHAN30 = 0

for i in range(len(file_tags)):
    file_tag = file_tags[i]
    data = read_csv(f'{filenames[i]}.csv')

    values = {'Original': [len(data[data[target] == NO]), len(data[data[target] == LESSTHAN30]),
                        len(data[data[target] == MORETHAN30])]}

    y = data.pop(target).values
    X = data.values
    labels = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # Train CSV
    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(f'data/train_and_test/{file_tag}_train.csv', index=False)

    # Test CSV
    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(f'data/train_and_test/{file_tag}_test.csv', index=False)
    values['Train'] = [len(np.delete(trnY, np.argwhere((trnY == LESSTHAN30) | (trnY == MORETHAN30)))),
                    len(np.delete(trnY, np.argwhere((trnY == NO) | (trnY == MORETHAN30)))),
                    len(np.delete(trnY, np.argwhere((trnY == NO) | (trnY == LESSTHAN30))))]
    values['Test'] = [len(np.delete(tstY, np.argwhere((tstY == LESSTHAN30) | (tstY == MORETHAN30)))),
                    len(np.delete(tstY, np.argwhere((tstY == NO) | (tstY == MORETHAN30)))),
                    len(np.delete(tstY, np.argwhere((tstY == NO) | (tstY == LESSTHAN30))))]

    figure(figsize=(12,4))
    multiple_bar_chart([NO, LESSTHAN30, MORETHAN30], values, title='Data distribution per dataset')
    savefig(f'images/{file_tag}_distribution_train_test.png')
    # show()