from pandas import read_csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart
from sklearn.model_selection import train_test_split
from pandas import concat, DataFrame


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
        file_names.append(file_name)
        file_paths.append(f'../classification/data/train_and_test/scaling/{file_name}')
print(file_paths)

target = 'readmitted'

for i in range(len(file_names)):
    file_name = file_names[i]
    file_path = file_paths[i]

    train = read_csv(f'{file_path}_train.csv')

    
    y = train['readmitted'].values
    y_train = pd.Index(y)
    target_count = y_train.value_counts()

    print("target count: ",target_count.values)
  

    #ind_positive_readmitted = target_count.index.get_loc(positive_readmitted)
    print('Readmitted 1 =', target_count.values[0])
    print('Readmitted 0 = ', target_count.values[1])
    print('Readmitted 2 = ', target_count.values[2])

    plt.clf()
    bar_chart(target_count.index, target_count.values, title='readmitted balance')
    savefig(f'images/balancing/{file_name}_readmitted_balance.png')



    readmitted_one = train[train['readmitted'] == 1]
    readmitted_zero = train[train['readmitted'] == 0]
    readmitted_two = train[train['readmitted'] == 2]





