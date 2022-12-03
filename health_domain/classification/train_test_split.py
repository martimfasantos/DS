import numpy as np
from pandas import read_csv, concat, unique, DataFrame
from matplotlib.pyplot import figure, savefig, show
from ds_charts import multiple_bar_chart
from sklearn.model_selection import train_test_split

file_tag = 'diabetic_data_drop_columns_mv'
data = read_csv('../data_preparation/data/diabetic_data_drop_columns_mv.csv')
target = 'readmitted'

NO = 'NO'
LESSTHAN30 = '<30'
MORETHAN30 = '>30'
values = {'Original': [len(data[data[target] == NO]), len(data[data[target] == LESSTHAN30]),
                       len(data[data[target] == MORETHAN30])]}

y = data.pop(target).values
X = data.values
labels = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

# Train CSV
train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'data/{file_tag}_train.csv', index=False)

# Test CSV
test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'data/{file_tag}_test.csv', index=False)
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