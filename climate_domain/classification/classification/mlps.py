from pandas import read_csv, DataFrame, unique
from matplotlib.pyplot import savefig, show, subplots, figure, plot, title
from ds_charts import multiple_line_chart, HEIGHT, plot_overfitting_study_mlp, plot_evaluation_results
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

file_tag = 'drought'
file_name = f'{file_tag}_under'
file_path = f'data/train_and_test/balancing/{file_name}'

target = 'class'

# Train 
train = read_csv(f'{file_path}_train.csv').sample(frac=1, random_state=8)
# print(train.shape)
trnY = train.pop(target).values
trnX = train.values
labels = unique(trnY)
labels.sort()

# Test
test = read_csv(f'data/train_and_test/balancing/{file_tag}_test.csv')
tstY = test.pop(target).values
tstX = test.values

# -------------- #
# MLP Study Adam #
# -------------- #

lr_type = ['constant'] #, 'invscaling', 'adaptive'] - only used if optimizer='sgd'
learning_rate = [.9, .6, .3, .1]
max_iter = [100, 150, 250, 500, 1000]
max_iter_warm_start = [max_iter[0]]
for el in max_iter[1:]:
    max_iter_warm_start.append(max_iter_warm_start[-1]+el)

measure = 'R2'
flag_pct = False
best = ('',  0, 0.0, '')
last_best = -10000
best_model = None
ncols = len(lr_type)

fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    tp = lr_type[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        warm_start = False
        for n in max_iter:
            pred = MLPClassifier(
                learning_rate=tp, learning_rate_init=lr, max_iter=n,
                activation='relu', warm_start=warm_start, verbose=False)
            pred.fit(trnX, trnY)
            prdY = pred.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            warm_start = True
            if yvalues[-1] > last_best:
                best = (tp, lr, n, 'adam')
                last_best = yvalues[-1]
                best_model = pred
        values[lr] = yvalues

    multiple_line_chart(
        max_iter_warm_start, values, ax=axs[0, k], title=f'MLP with lr_type={tp} and Adam solver', xlabel='mx iter', ylabel=measure, percentage=flag_pct)
savefig(f'images/mlps/{file_tag}_mlp_adam_study.png')
# show()
print(f'Adam: Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter ==> measure={last_best:.2f}')


# -------------- #
# MLPs Study SGD #
# -------------- #


lr_type = ['constant', 'invscaling', 'adaptive']
max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
learning_rate = [.1, .5, .9]

cols = len(lr_type)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    d = lr_type[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        for n in max_iter:
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                learning_rate_init=lr, max_iter=n, verbose=False, random_state=8)
            mlp.fit(trnX, trnY)
            prdY = mlp.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n, 'sgd')
                last_best = yvalues[-1]
                best_model = mlp
        values[lr] = yvalues
    multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d} and SGD solver',
                           xlabel='mx iter', ylabel='accuracy', percentage=True)
savefig(f'images/mlps/{file_tag}_mlp_sgd_study.png')
# show()
print(f'SGD: Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')


# -------- #
# Best MLP #
# -------- #

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images/mlps/{file_tag}_mlp_best.png')
# show()


# ----------- #
# Overfitting #
# ----------- #

y_tst_values = []
y_trn_values = []
warm_start = False
for n in max_iter:
    # print(f'MLP - lr type={best[1]} learning rate={best[0]} and nr_episodes={n}')
    MLPClassifier(
        learning_rate=best[0], learning_rate_init=best[1], max_iter=n,
        activation='relu', warm_start=warm_start, verbose=False)
    pred.fit(trnX, trnY)
    prd_tst_Y = pred.predict(tstX)
    prd_trn_Y = pred.predict(trnX)
    y_tst_values.append(accuracy_score(tstY, prdY))
    y_trn_values.append(accuracy_score(tstY, prdY))
    warm_start = True
plot_overfitting_study_mlp(max_iter, y_trn_values, y_tst_values, name=f'NN_{best[0]}_{best[1]}_{best[3]}', xlabel='nr episodes', ylabel=measure, pct=flag_pct)


# ------------- #
# Training Loss #
# ------------- #

lr_type = best[0]
lr = best[1]
solver = best[3]
best_model.fit(trnX, trnY)
prd_tst = best_model.predict(tstX)      
loss = best_model.loss_curve_
figure()
plot(loss)
title(f'Training loss NN_{lr_type}_{lr}_{solver}')
savefig(f'images/mlps/loss_NN_{lr_type}_{lr}_{solver}.png')