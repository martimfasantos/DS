import pandas as pd
from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import savefig, gca, title
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots

file_tag = 'diabetic_data'
file_path = f'data/feature_selection/{file_tag}_selected.csv'
data = read_csv(file_path)

data.pop('readmitted')
variables = data.columns.values


# ------- # 
#   PCA   #
# ------- # 

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)

pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_

# PLOT EXPLAINED VARIANCE RATIO
fig = figure(figsize=(20, 8))
title('Explained variance ratio')
xlabel('PC')
ylabel('ratio')
x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
bwidth = 0.7
ax = gca()
ax.set_xticklabels(x_values)
ax.set_ylim(0.0, 0.3)
ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
ax.plot(pca.explained_variance_ratio_)
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(i, v+0.02, f'{v*100:.1f}', ha='center', fontweight='bold')
savefig(f'images/feature_extraction/{file_tag}_explained_ratio_variance.png')

#show()

transf = pca.transform(data)

new_data = pd.DataFrame(data=transf)
new_data["readmitted"] = read_csv(f'data/feature_selection/{file_tag}_selected.csv')['readmitted']
new_data.to_csv(f'data/feature_extraction/{file_tag}_extracted.csv', index=False)
