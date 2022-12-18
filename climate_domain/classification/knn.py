import os
import sys
from pandas import DataFrame, read_csv, to_datetime, concat, unique
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from matplotlib.pyplot import figure, savefig, show, subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

# Parse terminal input
FLAG = ''
valid_flags = ('outliers', 'scaling', 'balancing', 'feature_selection', 'feature_extraction')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python train_test_split.py [outliers|scaling|balancing|feature_selection|feature_extraction]")
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

target = 'class'

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

    if (FLAG == 'balancing'):
        test = read_csv(f'data/train_and_test/balancing/drought_test.csv')
    else:
        test = read_csv(f'{file_path}_test.csv')
    tstY = test.pop(target).values
    tstX = test.values


    # ----- #
    #  KNN  #
    # ----- #

    eval_metric = accuracy_score
    nvalues = [1, 5, 9, 13, 17]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0

    for d in dist:
        y_tst_values = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
        # print(f'{file_name} - Accuracies using {d} distance: {y_tst_values}')
        values[d] = y_tst_values

    figure()
    multiple_line_chart(nvalues, values, title=f'KNN variants: {file_name}', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    savefig(f'../data_preparation/images/{FLAG}/knn/{file_name}_knn_study.png')
    # show()
    # print('Best results with %d neighbors and %s'%(best[0], best[1]))


    # -------------- #
    # Best KNN model #
    # -------------- #

    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(f'../data_preparation/images/{FLAG}/knn/{file_name}_knn_best.png')
    # show()


    # ----------------- #
    # Overfitting study #
    # ----------------- #

    def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
        savefig(f'../data_preparation/images/{FLAG}/knn/overfitting_{file_name}.png')

    _, d = best
    eval_metric = accuracy_score
    y_tst_values = []
    y_trn_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        prd_trn_Y = knn.predict(trnX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))