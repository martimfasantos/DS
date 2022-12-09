from pandas import DataFrame, read_csv, to_datetime, concat, unique
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from matplotlib.pyplot import figure, savefig, show, subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB


register_matplotlib_converters()
file_name = 'drought'
file_path = 'data/variables_encoding/drought_variables_encoding.csv'
data = read_csv(file_path, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
index_column = data.columns[0]
data = data.drop([index_column], axis = 1)
# print(data.describe())


# variables that have a Normal distribution
norm_dist_variables = ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
                       'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE' ]


def determine_outlier_thresholds(summary5, var, OPTION, OUTLIER_PARAM):
    # default parameter
    if OPTION == 'iqr':
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%'] + iqr
        bottom_threshold = summary5[var]['25%'] - iqr
    # for normal distribution
    elif OPTION == 'stdev':
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    else:
        raise ValueError('Unknown outlier parameter!')
    return top_threshold, bottom_threshold

numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

# Remove non numeric variables (ordinal but not numeric)
to_remove = ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ7']

for el in to_remove:
    if el in numeric_vars:
        numeric_vars.remove(el)

print('Original data:', data.shape)
summary5 = data.describe(include='number')


# ------------- #
# Drop outliers #
# ------------- #

STDEV_PARAM = 3
IQR_PARAM = 10

df = data.copy(deep=True)

for var in numeric_vars:
    if var in norm_dist_variables:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'stdev', STDEV_PARAM)
    else:
        top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
    outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
    df.drop(outliers.index, axis=0, inplace=True)
    # if var in norm_dist_variables:
    # print(f'{var} = {outliers.shape[0]}')
df.to_csv(f'data/outliers/{file_name}_drop_outliers.csv', index=True)
print('data after dropping outliers:', df.shape)


# ----------------- #
# Truncate outliers #
# ----------------- #

IQR_PARAM = 8

df = data.copy(deep=True)

for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', 8)
    df[var] = df[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)

df.to_csv(f'data/outliers/{file_name}_truncate_outliers.csv', index=True)
# print('data after truncating outliers:', df.describe())


# # ----------------- #
# # KNN Drop outliers #
# # ----------------- #

# target = 'class'

# df = read_csv(f'data/outliers/{file_name}_drop_outliers.csv')

# X = df.drop(columns=['class'])
# y = df['class'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# eval_metric = accuracy_score
# nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# dist = ['manhattan', 'euclidean', 'chebyshev']
# values = {}
# best = (0, '')
# last_best = 0
# for d in dist:
#     y_tst_values = []
#     for n in nvalues:
#         knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#         knn.fit(X_train, y_train)
#         prd_tst_Y = knn.predict(X_test)
#         y_tst_values.append(eval_metric(y_test, prd_tst_Y))
#         if y_tst_values[-1] > last_best:
#             best = (n, d)
#             last_best = y_tst_values[-1]
#     values[d] = y_tst_values

# figure()
# multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
# savefig(f'images/outliers/{file_name}_drop_outliers_knn_study.png')
# # show()
# #print('Best results with %d neighbors and %s '%(best[0], best[1]))


# # --------------------- #
# # KNN Truncate outliers #
# # --------------------- #

# df = read_csv(f'data/outliers/{file_name}_truncate_outliers.csv')

# X = df.drop(columns=['class'])
# y = df['class'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# eval_metric = accuracy_score
# nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# dist = ['manhattan', 'euclidean', 'chebyshev']
# values = {}
# best = (0, '')
# last_best = 0
# for d in dist:
#     y_tst_values = []
#     for n in nvalues:
#         knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#         knn.fit(X_train, y_train)
#         prd_tst_Y = knn.predict(X_test)
#         y_tst_values.append(eval_metric(y_test, prd_tst_Y))
#         if y_tst_values[-1] > last_best:
#             best = (n, d)
#             last_best = y_tst_values[-1]
#     values[d] = y_tst_values

# figure()
# multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
# savefig(f'images/outliers/{file_name}_truncate_outliers_knn_study.png')
# # show()
# # print('Best results with %d neighbors and %s '%(best[0], best[1]))


# # ---------------- #
# # NB Drop outliers #
# # ---------------- #

# df = read_csv(f'data/outliers/{file_name}_drop_outliers.csv')

# X = df.drop(columns=['class'])
# y = df['class'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# estimators = {'GaussianNB': GaussianNB(),
#             #'MultinomialNB': MultinomialNB(),
#             'BernoulliNB': BernoulliNB()
#             #'CategoricalNB': CategoricalNB
#             }

# xvalues = []
# yvalues = []
# for clf in estimators:
#     xvalues.append(clf)
#     estimators[clf].fit(X_train, y_train)
#     prdY = estimators[clf].predict(X_test)
#     yvalues.append(accuracy_score(y_test, prdY))

# figure()
# bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
# savefig(f'images/outliers/{file_name}_drop_outliers_nb_study.png')
# # show()

# # labels = unique(y_train)

# # clf = GaussianNB()
# # clf.fit(X_train, y_train)
# # prd_trn = clf.predict(X_train)
# # prd_tst = clf.predict(X_test)
# # plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
# # savefig(f'images/outliers/{file_name}_drop_outliers_nb_best.png')
# # # show()

# # -------------------- #
# # NB Truncate outliers #
# # -------------------- #

# df = read_csv(f'data/outliers/{file_name}_truncate_outliers.csv')

# X = df.drop(columns=['class'])
# y = df['class'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# estimators = {'GaussianNB': GaussianNB(),
#             #'MultinomialNB': MultinomialNB(),
#             'BernoulliNB': BernoulliNB()
#             #'CategoricalNB': CategoricalNB
#             }

# xvalues = []
# yvalues = []
# for clf in estimators:
#     xvalues.append(clf)
#     estimators[clf].fit(X_train, y_train)
#     prdY = estimators[clf].predict(X_test)
#     yvalues.append(accuracy_score(y_test, prdY))

# figure()
# bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
# savefig(f'images/outliers/{file_name}_truncate_outliers_nb_study.png')
# # show()

# # labels = unique(y_train)

# # clf = GaussianNB()
# # clf.fit(X_train, y_train)
# # prd_trn = clf.predict(X_train)
# # prd_tst = clf.predict(X_test)
# # plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
# # savefig(f'images/outliers/{file_name}_truncate_outliers_nb_best.png')
# # # show()
