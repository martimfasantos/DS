import os
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score

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
        filenames.append(f'data/train_and_test/{file_tag}')

# print(file_tags)
# print(filenames)

target = 'readmitted'
for i in range(len(file_tags)):
    file_tag = file_tags[i]

    train = read_csv(f'{filenames[i]}_train.csv')
    trnY = train.pop(target).values
    trnX = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filenames[i]}_test.csv')
    tstY = test.pop(target).values
    tstX = test.values
    print('CHECKPOINT 1')
    # ----- #
    #  KNN  #
    # ----- #

    eval_metric = accuracy_score
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0

    for d in dist:
        y_tst_values = []
        for n in nvalues:
            print('K CHECKPOINT BEGIN: ' + str(n))
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
            print('K CHECKPOINT END: ' + str(n))
        print(y_tst_values)
        values[d] = y_tst_values
    print('CHECKPOINT 2')

    figure()
    multiple_line_chart(nvalues, values, title=f'KNN variants: {file_tag}', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    savefig(f'images/knn/{file_tag}_knn_study.png')
    #show()
    # print('Best results with %d neighbors and %s'%(best[0], best[1]))
    print('CHECKPOINT 3')

    # -------------- #
    # Best KNN model #
    # -------------- #

    # clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    # clf.fit(trnX, trnY)
    # prd_trn = clf.predict(trnX)
    # prd_tst = clf.predict(tstX)
    # plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    # savefig(f'images/knn/{file_tag}_knn_best.png')
    # # show()

    # print('CHECKPOINT 4')
    # # ----------------- #
    # # Overfitting study #
    # # ----------------- #

    # def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    #     evals = {'Train': prd_trn, 'Test': prd_tst}
    #     figure()
    #     multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    #     savefig(f'images/overfitting_{name}.png')

    # _, d = best
    # eval_metric = accuracy_score
    # y_tst_values = []
    # y_trn_values = []
    # for n in nvalues:
    #     knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    #     knn.fit(trnX, trnY)
    #     prd_tst_Y = knn.predict(tstX)
    #     prd_trn_Y = knn.predict(trnX)
    #     y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    #     y_trn_values.append(eval_metric(trnY, prd_trn_Y))
    # plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))
    # print('CHECKPOINT 5')