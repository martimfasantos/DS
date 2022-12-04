from pandas import DataFrame, read_csv, to_datetime, concat, unique
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from matplotlib.pyplot import figure, savefig, show, subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

register_matplotlib_converters()
data = read_csv('../datasets/classification/drought.csv', na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'])

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]


# --------------------- #
# Z-score Normalization #
# --------------------- #

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_zscore.to_csv('data/_scaled_zscore.csv', index=False)



# --------------------- #
# MinMax normalization  #
# --------------------- #

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_minmax.to_csv('data/_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())



# ---------- #
# Comparison #
# ---------- #

fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
savefig('./images/boxplot_scalling.png')
#show()


# ------------------ #
#     KNN Min Max    #
# ------------------ #

filename = 'classification/data'
target = 'class'

df = read_csv('data/_scaled_minmax.csv')


X = df.drop(columns=['class'])
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


eval_metric = accuracy_score
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    y_tst_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(X_train, y_train)
        prd_tst_Y = knn.predict(X_test)
        y_tst_values.append(eval_metric(y_test, prd_tst_Y))
        if y_tst_values[-1] > last_best:
            best = (n, d)
            last_best = y_tst_values[-1]
    values[d] = y_tst_values

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig('images/climate_knn_minmax_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))



# ------------------ #
#     KNN ZScore     #
# ------------------ #

filename = 'classification/data'
target = 'class'

df = read_csv('data/_scaled_zscore.csv')


X = df.drop(columns=['class'])
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


eval_metric = accuracy_score
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0
for d in dist:
    y_tst_values = []
    for n in nvalues:
        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
        knn.fit(X_train, y_train)
        prd_tst_Y = knn.predict(X_test)
        y_tst_values.append(eval_metric(y_test, prd_tst_Y))
        if y_tst_values[-1] > last_best:
            best = (n, d)
            last_best = y_tst_values[-1]
    values[d] = y_tst_values

figure()
multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
savefig('images/climate_knn_zscore_study.png')
show()
print('Best results with %d neighbors and %s'%(best[0], best[1]))


labels = unique(y_train)
clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
clf.fit(X_train, y_train)
prd_trn = clf.predict(X_train)
prd_tst = clf.predict(X_test)
plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
savefig('images/climate_knn_best.png')
show()


def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting climate', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig('images/overfitting_climate.png')

d = 'euclidean'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in nvalues:
    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(X_train, y_train)
    prd_tst_Y = knn.predict(X_test)
    prd_trn_Y = knn.predict(X_train)
    y_tst_values.append(eval_metric(y_test, prd_tst_Y))
    y_trn_values.append(eval_metric(y_train, prd_trn_Y))
plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))




# ----------------------------- #
#     Naive Bayes with MinMax   #
# ----------------------------- #

df = read_csv('data/_scaled_minmax.csv')


X = df.drop(columns=['class'])
y = df['class'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


labels = unique(y_train)


clf = GaussianNB()
clf.fit(X_train, y_train)
prd_trn = clf.predict(X_train)
prd_tst = clf.predict(X_test)
plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
savefig('images/climate_nb_best.png')
show()


estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(X_train, y_train)
    prdY = estimators[clf].predict(X_test)
    yvalues.append(accuracy_score(y_test, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/climate_nb_minmax_study.png')
show()

# ----------------------------- #
#     Naive Bayes with Zscore   #
# ----------------------------- #

df = read_csv('data/_scaled_zscore.csv')


X = df.drop(columns=['class'])
y = df['class'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


labels = unique(y_train)


clf = GaussianNB()
clf.fit(X_train, y_train)
prd_trn = clf.predict(X_train)
prd_tst = clf.predict(X_test)
plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
savefig('images/climate_nb_best.png')
show()


estimators = {'GaussianNB': GaussianNB(),
              #'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(X_train, y_train)
    prdY = estimators[clf].predict(X_test)
    yvalues.append(accuracy_score(y_test, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'images/climate_nb_zscore_study.png')
show()

