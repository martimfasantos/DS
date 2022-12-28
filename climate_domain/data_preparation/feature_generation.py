import pandas as pd
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots
from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import savefig, gca, title

file_tag = 'drought'
file_path = f'data/feature_extraction/{file_tag}_extracted.csv'
data = read_csv(file_path)

original_file_tag = 'drought'
original_file_path = f'data/outliers/{file_tag}_drop_outliers.csv'
original_data = read_csv(original_file_path)

date = original_data['date']

data['season_cos'] = 0
data['season_sin'] = 0

for ind in date.index:
    date_str = str(date[ind])
    if len(date_str) == len('4012000'):
        day = int(date_str[0])
        mon = int(date_str[1:3])
        yer = int(date_str[3:])
    else:
        day = int(date_str[0:2])
        mon = int(date_str[2:4])
        yer = int(date_str[4:])
    # winter
    if mon == 1 or mon == 2:
        data['season_cos'][ind] = 0
        data['season_sin'][ind] = 1
    elif mon == 3:
        # winter
        if day < 20:
            data['season_cos'][ind] = 0
            data['season_sin'][ind] = 1
        # spring
        else:
            data['season_cos'][ind] = 1
            data['season_sin'][ind] = 0
    # spring
    elif mon == 4 or mon == 5:
        data['season_cos'][ind] = 1
        data['season_sin'][ind] = 0
    elif mon == 6:
        # spring
        if day < 21:
            data['season_cos'][ind] = 1
            data['season_sin'][ind] = 0
        # summer
        else:
            data['season_cos'][ind] = 0
            data['season_sin'][ind] = -1
    # summer
    elif mon == 7 or mon == 8:
        data['season_cos'][ind] = 0
        data['season_sin'][ind] = -1
    elif mon == 9:
        # summer
        if day < 23:
            data['season_cos'][ind] = 0
            data['season_sin'][ind] = -1
        # fall
        else:
            data['season_cos'][ind] = -1
            data['season_sin'][ind] = 0       
    # fall
    elif mon == 10 or mon == 11:
        data['season_cos'][ind] = -1
        data['season_sin'][ind] = 0
    elif mon == 12:
        # fall
        if day < 22:
            data['season_cos'][ind] = -1
            data['season_sin'][ind] = 0
        # winter
        else:
            data['season_cos'][ind] = 0
            data['season_sin'][ind] = 1  
    else:
        print("erro")

data.to_csv(f'data/feature_generation/{file_tag}_generated.csv', index=False)
