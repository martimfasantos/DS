import os
import sys
import numpy as np
from pandas import read_csv, unique
from matplotlib.pyplot import figure, show, savefig
from ds_charts import plot_evaluation_results, bar_chart, plot_evaluation_results_ternary
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score


# Parse terminal input
FLAG = ''
valid_flags = ('missing_values', 'outliers', 'scaling', 'balancing', 'feature_selection', 'feature_extraction')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python naive_bayes.py [missing_values|outliers|scaling|balancing|feature_selection|feature_extraction]")
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
        file_paths.append(f'data/train_and_test/{FLAG}/{file_name}')
# print(file_paths)

target = 'readmitted'

for i in range(len(file_names)):
        file_name = file_names[i]
        file_path = file_paths[i]

        # Train 
        if (FLAG == 'balancing'):
            train = read_csv(f'{file_path}.csv')
        else:
            train = read_csv(f'{file_path}_train.csv')
        trnY = train.pop(target).values
        trnX = train.values
        labels = unique(trnY)
        labels.sort()

        # Test
        if (FLAG == 'balancing'):
            test = read_csv(f'data/train_and_test/balancing/diabetic_data_test.csv')
        else:
            test = read_csv(f'{file_path}_test.csv')
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
            try:
                estimators[clf].fit(trnX, trnY)
                prdY = estimators[clf].predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
            except:
                xvalues.remove(clf)

        figure()
        bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        savefig(f'../data_preparation/images/{FLAG}/naive_bayes/{file_name}_nb_study.png')

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
        plot_evaluation_results_ternary(['NO', '<30', '>30'], trnY, prd_trn, tstY, prd_tst)
        savefig(f'../data_preparation/images/{FLAG}/naive_bayes/{file_name}_nb_best.png')
        # show()