import numpy as np
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from ds_charts import bar_chart, multiple_line_chart, plot_overfitting_study, plot_evaluation_results
from sklearn.metrics import accuracy_score


# Best option
file_tag = 'drought'
file_path = f'data/train_and_test/scaling/{file_tag}_scaled_zscore'

target = 'class'

# Train 
train = read_csv(f'{file_path}_train.csv')
unnamed_column = train.columns[0]
train = train.drop([unnamed_column], axis=1)
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'{file_path}_test.csv')
unnamed_column = test.columns[0]
test = test.drop([unnamed_column], axis=1)
tstY = test.pop(target).values
tstX = test.values


# ----- #
#  KNN  #
# ----- #

eval_metric = accuracy_score
nvalues = [1, 5, 15, 25, 50, 100]
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
    # print(f'{file_tag} - Accuracies using {d} distance: {y_tst_values}')
    values[d] = y_tst_values

figure()
multiple_line_chart(nvalues, values, title=f'KNN variants: {file_tag}', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig(f'../data_preparation/images/best_results/knn/{file_tag}_knn_study.png')
# show()
# print('Best results with %d neighbors and %s'%(best[0], best[1]))
print("KNN 1")


# -------------- #
# Best KNN model #
# -------------- #

clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'../data_preparation/images/best_results/knn/{file_tag}_knn_best.png')
# show()

print("KNN 2")


# ------------------------------ #
# Neighbours near best KNN model #
# ------------------------------ #

eval_metric = accuracy_score
nvalues_int = list(range(best[0]-5, best[0]+6))
dist = ['manhattan', 'euclidean', 'chebyshev']
values_int = {}
best = (0, '')
last_best = 0

for d in dist:
    y_tst_values = []
    for n in nvalues_int:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(trnX, trnY)
        prd_tst_Y = knn.predict(tstX)
        y_tst_values.append(eval_metric(tstY, prd_tst_Y))
        if y_tst_values[-1] > last_best and n < 100:
            best = (n, d)
            last_best = y_tst_values[-1]
    # print(f'{file_tag} - Accuracies using {d} distance: {y_tst_values}')
    values_int[d] = y_tst_values

figure()
multiple_line_chart(nvalues_int, values_int, title=f'KNN variants: {file_tag}', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig(f'../data_preparation/images/best_results/knn/{file_tag}_knn_study_interval.png')

print("KNN 3")


# -------------------------------- #
# All Distances for best KNN Model #
# -------------------------------- #

k, _ = best
eval_metric = accuracy_score
dist = ['manhattan', 'euclidean', 'chebyshev']
y_values = []

for d in dist:
    knn = KNeighborsClassifier(n_neighbors=k, metric=d)
    knn.fit(trnX, trnY)
    prd_tst_Y = knn.predict(tstX)
    y_values.append(eval_metric(tstY, prd_tst_Y))

figure()
bar_chart(dist, y_values, title=f'Comparison of distances for KNN_K={k}', ylabel='accuracy', percentage=True)
savefig(f'../data_preparation/images/best_results/knn/{file_tag}_knn_best_distances.png')
# show()

print("KNN 4")

# ----------------- #
# Overfitting study #
# ----------------- #

def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'../data_preparation/images/best_results/knn/overfitting_{file_tag}.png')

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

print("Overfitting")


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
savefig(f'../data_preparation/images/best_results/naive_bayes/{file_tag}_nb_study.png')
# show()

print("NB 1")


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
savefig(f'../data_preparation/images/best_results/naive_bayes/{file_tag}_nb_best.png')
# show()

print("NB 2")