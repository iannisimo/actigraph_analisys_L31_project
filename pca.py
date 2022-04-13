# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')

from matplotlib import projections

from operator import le
from utils import load_dataset, tables, fix_inline
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

fix_inline()
(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(tables.AHA, tables.METADATA, length=300, overlap=120, class_method='macs')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30)
n_features = X_train.shape[1]
n_samples = X_train.shape[0]

# %%

pca = KernelPCA(n_jobs=-1)

pca_fitted = pca.fit(X_train)
x_train_kern = pca_fitted.transform(X_train)

mdl = BernoulliNB()

scores = cross_val_score(mdl, X_train, y_train, cv=5, n_jobs=-1)
print(scores, np.mean(scores))

scores_kern = cross_val_score(mdl, x_train_kern, y_train, cv=5, n_jobs=-1)
print(scores_kern, np.mean(scores_kern))

# %%
fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(x_train_kern[:, 0], x_train_kern[:, 1], c=y_train)
ax.set_ylabel("Principal component #1")
ax.set_xlabel("Principal component #0")
ax.set_title("Projection of testing data\n using PCA")
plt.show()
# %%
