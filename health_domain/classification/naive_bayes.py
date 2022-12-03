import os
import numpy as np
from pandas import read_csv, unique
from matplotlib.pyplot import figure, show, savefig
from ds_charts import plot_evaluation_results, bar_chart
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

# Folder path
dir_path = '../data_preparation/data/'

# List to store files
file_tags = []
filenames = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_tag = os.path.splitext(file)[0]
        file_tags.append(file_tag)
        filenames.append(f'data/{file_tag}')

# print(file_tags)
# print(filenames)

file_tag = 'diabetic_data_drop_columns_dv'
filename = 'data/diabetic_data_drop_columns_mv'
target = 'readmitted'

# Train
train = read_csv(f'{filename}_train.csv')
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'{filename}_test.csv')
tstY = test.pop(target).values
tstX = test.values
 
# ----------- #
# Naive Bayes #
# ----------- #

# Comparison of Naive Bayes Models
estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/{file_tag}_nb_study.png')
# show()


# ------------- #
# Best NB model #
# ------------- #

max = np.max(yvalues)
max_index = yvalues.index(max)
best_model = estimators[xvalues[max_index]]

# print(best_model)

clf = best_model
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/{file_tag}_nb_best.png')
# show()
