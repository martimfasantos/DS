from pandas import DataFrame, read_csv, to_datetime, concat, unique
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, plot_evaluation_results, multiple_line_chart, plot_overfitting_study, plot_evaluation_results_ternary
from matplotlib.pyplot import figure, savefig, show, subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score
import numpy as np

register_matplotlib_converters()

# Choose the BEST MODEL based on results from KNN and NB:
# -> Variable encoding 1
# -> Drop columns + most frequent substitution for missing values
file_name = 'diabetic_data_drop_outliers'
file_tag = 'diabetic_data'
file_path = 'data/outliers/diabetic_data_drop_outliers.csv'
data = read_csv(file_path, na_values='?')
index_column = data.columns[0]
data = data.drop([index_column], axis=1)

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']
# No Date variables in this dataset

# Since they are unique keys, it does not make sense to consider
# them in the normalization procedure
numeric_vars.remove('patient_nbr')
numeric_vars.remove('encounter_id')

# Variables that do not require normalization for better results
to_remove = ['metformin_variation', 'repaglinide_variation', 'nateglinide_variation', 
             'chlorpropamide_variation', 'glimepiride_variation', 'glipizide_variation', 
             'glyburide_variation', 'pioglitazone_variation', 'rosiglitazone_variation', 
             'acarbose_variation', 'miglitol_variation', 'examide_prescribed', 'examide_variation', 
             'citoglipton_prescribed', 'citoglipton_variation', 'insulin_variation', 
             'glyburide-metformin_variation', 'acetohexamide_variation', 'tolbutamide_variation',
             'troglitazone_variation', 'glipizide-metformin_variation', 'glimepiride-pioglitazone_variation', 
             'metformin-rosiglitazone_variation', 'metformin-pioglitazone_variation'
             'max_glu_serum_level', 'A1Cresult_level', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
             'metformin-pioglitazone_variation', 'max_glu_serum_level']

for el in to_remove:
    if el in numeric_vars:
        numeric_vars.remove(el)

# Remove class (readmitted) from numeric vars
numeric_vars.remove('readmitted')

df_num = data[numeric_vars]
df_symb = data[symbolic_vars]
df_bool = data[boolean_vars]
df_target = data['readmitted']


# --------------------- #
# Z-score Normalization #
# --------------------- #

# scale numeric variables and concat to the rest to create a new csv file
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
temp_norm_data_zscore = concat([tmp, df_symb, df_bool], axis=1)
norm_data_zscore = concat([temp_norm_data_zscore, df_target], axis=1)
norm_data_zscore.to_csv(f'data/scaling/{file_name}_scaled_zscore.csv', index=False)
# print(norm_data_zscore.describe())


# --------------------- #
# MinMax normalization  #
# --------------------- #

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=data.index, columns= numeric_vars)
temp_norm_data_minmax = concat([tmp, df_symb, df_bool], axis=1)
norm_data_minmax = concat([temp_norm_data_minmax, df_target], axis=1)
norm_data_minmax.to_csv(f'data/scaling/{file_name}_scaled_minmax.csv', index=False)
# print(norm_data_minmax.describe())


# ---------- #
# Comparison #
# ---------- #

fig, axs = subplots(1, 3, figsize=(50,15),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0], rot=90)
axs[0, 1].set_title('Z-score normalization')
temp_norm_data_zscore.boxplot(ax=axs[0, 1], rot=90)
axs[0, 2].set_title('MinMax normalization')
temp_norm_data_minmax.boxplot(ax=axs[0, 2], rot=90)
savefig(f'images/scaling/{file_tag}_scale_comparison.png')
# show()


# # ------------------ #
# #         KNN        #
# # ------------------ #

# print("KNN NO SCALING BEGIN")

# file_path = 'data/outliers/diabetic_data_drop_outliers.csv'
# df = read_csv(file_path)

# X = df.drop(columns=['readmitted'])
# y = df['readmitted'].values

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
#         print("     VALUE=" + str(n))
#     print(" METRIC=" + d)
#     values[d] = y_tst_values

# figure()
# multiple_line_chart(nvalues, values, title='KNN Scaling Variants: NO SCALING', xlabel='n', ylabel=str(accuracy_score), percentage=True)
# savefig(f'images/scaling/{file_tag}_knn_no_scaling_study.png')
# print("KNN NO SCALING END")

# # ------------------ #
# #     KNN Min Max    #
# # ------------------ #

# print("KNN MIN-MAX BEGIN")

# df = read_csv(f'data/scaling/{file_name}_scaled_minmax.csv')

# X = df.drop(columns=['readmitted'])
# y = df['readmitted'].values

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
#         print("     VALUE=" + str(n))
#     print(" METRIC=" + d)
#     values[d] = y_tst_values

# figure()
# multiple_line_chart(nvalues, values, title='KNN Scaling Variants: MIN-MAX SCALING', xlabel='n', ylabel=str(accuracy_score), percentage=True)
# savefig(f'images/scaling/{file_tag}_knn_minmax_study.png')
# # show()
# # print('Best results with %d neighbors and %s'%(best[0], best[1]))

# print("KNN MIN-MAX END")

# # ------------------ #
# #     KNN ZScore     #
# # ------------------ #

# print("KNN Z-SCORE BEGIN")
# df = read_csv(f'data/scaling/{file_name}_scaled_zscore.csv')

# X = df.drop(columns=['readmitted'])
# y = df['readmitted'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# eval_metric = accuracy_score
# nvalues = [1, 5, 9, 13, 17]
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
#         print("     VALUE=" + str(n))
#     print(" METRIC=" + d)
#     values[d] = y_tst_values

# figure()
# multiple_line_chart(nvalues, values, title='KNN Scaling Variants: Z-SCORE SCALING', xlabel='n', ylabel=str(accuracy_score), percentage=True)
# savefig(f'images/scaling/{file_tag}_knn_zscore_study.png')
# # show()
# # print('Best results with %d neighbors and %s'%(best[0], best[1]))
# print("KNN Z-SCORE END")

# # # ---------------- #
# # #     Best KNN     #
# # # ---------------- #

# print("BEST KNN BEGIN")

# labels = unique(y_train)
# clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
# clf.fit(X_train, y_train)
# prd_trn = clf.predict(X_train)
# prd_tst = clf.predict(X_test)
# plot_evaluation_results_ternary(labels, y_train, prd_trn, y_test, prd_tst)
# savefig(f'images/scaling/{file_tag}_knn_best.png')
# # show()

# print("BEST KNN END")

# # ----------------------- #
# #     Overfitting KNN     #
# # ----------------------- #

# print("OVERFITTING KNN BEGIN")

# def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
#     evals = {'Train': prd_trn, 'Test': prd_tst}
#     figure()
#     multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {file_name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
#     savefig(f'images/scaling/overfitting_{file_name}.png')

# _, d = best
# eval_metric = accuracy_score
# y_tst_values = []
# y_trn_values = []
# for n in nvalues:
#     knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#     knn.fit(X_train, y_train)
#     prd_tst_Y = knn.predict(X_test)
#     prd_trn_Y = knn.predict(X_train)
#     y_tst_values.append(eval_metric(y_test, prd_tst_Y))
#     y_trn_values.append(eval_metric(y_train, prd_trn_Y))
#     print("     VALUE=" + str(n))
# plot_overfitting_study(nvalues, y_trn_values, y_tst_values, name=f'KNN_K={n}_{d}', xlabel='K', ylabel=str(eval_metric))
# print("OVERFITTING KNN END")

# ----------------------------- #
#           Naive Bayes         #
# ----------------------------- #

print("NAIVE BAYES BEGIN")
file_path = 'data/outliers/diabetic_data_drop_outliers.csv'
df = read_csv(file_path)

X = df.drop(columns=['readmitted'])
y = df['readmitted'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

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
bar_chart(xvalues, yvalues, title='NB Scaling Variants: NO SCALING', ylabel='accuracy', percentage=True)
savefig(f'images/scaling/{file_tag}_nb_no_scaling_study.png')


# ----------------------------- #
#     Naive Bayes with MinMax   #
# ----------------------------- #

df = read_csv(f'data/scaling/{file_name}_scaled_minmax.csv')

X = df.drop(columns=['readmitted'])
y = df['readmitted'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# NOT BINARY CLASSIFICATION
labels = unique(y_train)

max = np.max(yvalues)
max_index = yvalues.index(max)
best_model = estimators[xvalues[max_index]]

# print(best_model)

clf = best_model
clf.fit(X_train, y_train)
prd_trn = clf.predict(X_train)
prd_tst = clf.predict(X_test)
plot_evaluation_results_ternary(labels, y_train, prd_trn, y_test, prd_tst)
savefig(f'images/scaling/{file_tag}_nb_minmax_best.png')
# show()
# show()

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
bar_chart(xvalues, yvalues, title='NB Scaling Variants: MIN-MAX SCALING', ylabel='accuracy', percentage=True)
savefig(f'images/scaling/{file_tag}_nb_minmax_study.png')
# show()


# ----------------------------- #
#     Naive Bayes with Zscore   #
# ----------------------------- #

df = read_csv(f'data/scaling/{file_name}_scaled_zscore.csv')

X = df.drop(columns=['readmitted'])
y = df['readmitted'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=8)

# NOT BINARY CLASSIFICATION
labels = unique(y_train)

max = np.max(yvalues)
max_index = yvalues.index(max)
best_model = estimators[xvalues[max_index]]

# print(best_model)

clf = best_model
clf.fit(X_train, y_train)
prd_trn = clf.predict(X_train)
prd_tst = clf.predict(X_test)
plot_evaluation_results_ternary(labels, y_train, prd_trn, y_test, prd_tst)
savefig(f'images/scaling/{file_tag}_nb_zscore_best.png')
# show()

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
bar_chart(xvalues, yvalues, title='NB Scaling Variants: Z-SCORE SCALING', ylabel='accuracy', percentage=True)
savefig(f'images/scaling/{file_tag}_nb_zscore_study.png')
# show()

print("NAIVE BAYES END")