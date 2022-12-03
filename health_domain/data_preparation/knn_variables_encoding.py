import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score
    
for f in (1, 2):
    print(f)
    file_tag = 'diabetic_data'
    filename = 'data/diabetic_data_variables_encoding_' + str(f) + '.csv'
    print(filename)
    data: DataFrame = read_csv(filename, na_values='?')
    target = 'readmitted'

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=41)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(f'data/{file_tag}_train_variables_encoding_{str(f)}.csv', index=False)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(f'data/{file_tag}_test_variables_encoding_{str(f)}.csv', index=False)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    file_tag = 'diabetic_data'
    filename = 'data/diabetic_data'
    target = 'readmitted'

    train: DataFrame = read_csv(f'{filename}_train_variables_encoding_{str(f)}.csv')
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename}_test_variables_encoding_{str(f)}.csv')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

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
            knn.fit(trnX, trnY)
            prd_tst_Y = knn.predict(tstX)
            y_tst_values.append(eval_metric(tstY, prd_tst_Y))
            if y_tst_values[-1] > last_best:
                best = (n, d)
                last_best = y_tst_values[-1]
        values[d] = y_tst_values
        print(y_tst_values)
        print(sum(y_tst_values) / len(y_tst_values))
        
    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel=str(accuracy_score), percentage=True)
    savefig('images/diabetic_data_knn_study_variables_encoding_' + str(f) + '.png')
