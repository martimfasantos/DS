from numpy import argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.tree import DecisionTreeClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, plot_overfitting_study_dt
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from subprocess import call


file_tag = 'drought'
file_name = f'{file_tag}_under'
file_path = f'data/train_and_test/balancing/{file_name}'

target = 'class'

# Train 
train = read_csv(f'{file_path}_train.csv')
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'data/train_and_test/balancing/drought_test.csv')
tstY = test.pop(target).values
tstX = test.values


# -------------------- #
# Random Decision Tree #
# -------------------- #

min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
max_depths = [2, 5, 10, 15, 20, 25]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
last_best = 0
best_model = None

figure()
fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for imp in min_impurity_decrease:
            tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                last_best = yvalues[-1]
                best_model = tree

        values[d] = yvalues
    multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                        xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
savefig(f'images/decision_trees/{file_tag}_dt_study.png')
# show()
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))


# --------- #
# Best Tree #
# --------- #

file_tree = f'images/decision_trees/{file_tag}_dt_best_tree.png'
dot_data = export_graphviz(best_model, out_file=f'images/decision_trees/{file_tag}_dt_best_tree.dot', filled=True, rounded=True, special_characters=True)

# Convert to png
call(['dot', '-Tpng', f'images/decision_trees/{file_tag}_dt_best_tree.dot', '-o', file_tree, '-Gdpi=600'])


# ------------------ #
# Best Decision Tree #
# ------------------ #

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/decision_trees/{file_tag}_dt_best.png')
# show()


# ------------------- #
# Features importance #
# ------------------- #

variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
savefig(f'images/decision_trees/{file_tag}_dt_ranking.png')


# ----------- #
# Overfitting #
# ----------- #

imp = 0.0001
f = 'entropy'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for d in max_depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    prd_tst_Y = tree.predict(tstX)
    prd_trn_Y = tree.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study_dt(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))