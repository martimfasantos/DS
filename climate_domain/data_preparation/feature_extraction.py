from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots

data: DataFrame = read_csv('data/feature_selection/drought_selected.csv')
data.pop('class')
data.pop('date     ')
variables = data.columns.values

# # # # # # 
#   PCA   #
# # # # # # 

from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import gca, title
import pandas as pd

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)

pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_

# PLOT EXPLAINED VARIANCE RATIO
fig = figure(figsize=(4, 4))
title('Explained variance ratio')
xlabel('PC')
ylabel('ratio')
x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
bwidth = 0.5
ax = gca()
ax.set_xticklabels(x_values)
ax.set_ylim(0.0, 1.0)
ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
ax.plot(pca.explained_variance_ratio_)
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(i, v+0.05, f'{v*100:.1f}', ha='center', fontweight='bold')
#show()

transf = pca.transform(data)

new_data = pd.DataFrame(data=transf)
new_data["class"] = read_csv('data/feature_selection/drought_selected.csv')['class']
new_data["date"] = read_csv('data/feature_selection/drought_selected.csv')['date     ']
new_data.to_csv(f'data/feature_extraction/drought_extracted.csv', index=False)
