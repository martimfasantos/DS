from numpy import std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import plot_evaluation_results_ternary, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study_gb

file_tag = 'diabetic_data'
file_name = f'{file_tag}_under'
file_path = f'data/train_and_test/balancing/{file_name}'

target = 'readmitted'

# Train 
train = read_csv(f'{file_path}_train.csv')
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'data/train_and_test/balancing/{file_tag}_test.csv')
tstY = test.pop(target).values
tstX = test.values


# ----------------------- #
# Gradient Boosting Study #
# ----------------------- #

n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
max_depths = [5, 10, 25]
learning_rate = [.1, .5, .9]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        for n in n_estimators:
            gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
            gb.fit(trnX, trnY)
            prdY = gb.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = gb
        values[lr] = yvalues
    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                           xlabel='nr estimators', ylabel='accuracy', percentage=True)
savefig(f'images/gradient_boosting/{file_tag}_gb_study.png')
# show()
print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))


# ---------------------- #
# Best Gradient Boosting #
# ---------------------- #

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results_ternary(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/gradient_boosting/{file_tag}_gb_best.png')
# show()


# ------------------- #
# Features importance #
# ------------------- #

variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
stdevs = std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
elems = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/gradient_boosting/{file_tag}_gb_ranking.png')


# ----------- #
# Overfitting #
# ----------- #

lr = 0.7
max_depth = 10
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in n_estimators:
    gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
    gb.fit(trnX, trnY)
    prd_tst_Y = gb.predict(tstX)
    prd_trn_Y = gb.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study_gb(n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}', xlabel='nr_estimators', ylabel=str(eval_metric))
