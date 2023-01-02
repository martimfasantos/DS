from numpy import random
from pandas import DataFrame, read_csv, unique, Series
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results_ternary, multiple_line_chart, HEIGHT
from sklearn.metrics import accuracy_score
from ds_charts import plot_overfitting_study_mlp

file_tag = 'diabetic_data'
file_name = f'{file_tag}_under'
file_path = f'data/train_and_test/balancing/{file_name}'

target = 'readmitted'

# Train 
train = read_csv(f'{file_path}_train.csv').sample(frac=0.25, random_state=8)
print(train.shape)
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'data/train_and_test/balancing/{file_tag}_test.csv')
tstY = test.pop(target).values
tstX = test.values


# ---------- #
# MLPs Study #
# ---------- #

lr_type = ['constant', 'invscaling', 'adaptive']
max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
learning_rate = [.1, .5, .9]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(lr_type)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    d = lr_type[k]
    print(f'- {d}')
    values = {}
    for lr in learning_rate:
        print(f'\t{lr}')
        yvalues = []
        for n in max_iter:
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                learning_rate_init=lr, max_iter=n, verbose=False, random_state=8)
            mlp.fit(trnX, trnY)
            prdY = mlp.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = mlp
            print(f'\t\t{n}')
        values[lr] = yvalues
    multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                           xlabel='mx iter', ylabel='accuracy', percentage=True)
savefig(f'images/mlps/{file_tag}_mlp_study.png')
# show()
print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')


# -------- #
# Best MLP #
# -------- #

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results_ternary(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/mlps/{file_tag}_mlp_best.png')
#show()


# ----------- #
# Overfitting # TODO change to other settings
# ----------- #

lr_type = 'adaptive'
lr = 0.9
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for n in max_iter:
    mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
    mlp.fit(trnX, trnY)
    prd_tst_Y = mlp.predict(tstX)
    prd_trn_Y = mlp.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
plot_overfitting_study_mlp(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr episodes', ylabel=str(eval_metric))