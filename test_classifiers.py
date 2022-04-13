# %%
import os
from termcolor import colored, cprint
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables, fix_inline
from util_classes import sg_cross_val_score, latex_cross_val
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

fix_inline()


(X_train, y_train, g_train, _, _, _, _) = load_dataset(
    tables.AHA, tables.METADATA, length=360, overlap=-180, 
    class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
)

# (X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
#     tables.WEEK, tables.METADATA, length=86400, overlap=86400, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )


# %%
cols = 1 if len(y_train.shape) == 1 else y_train.shape[1]
for col in range(0,cols):
    cprint(f'\n\n\tCurrent Column\t{col}\n\n', 'green')
    y_col = y_train[:,col] if cols > 1 else y_train
    mdls = [
        # SGDClassifier(),
        # Perceptron(),
        # LinearSVC(),
        # NuSVC(nu=0.01, probability=True),
        SVC(probability=True),
        # KNeighborsClassifier(),
        # GaussianProcessClassifier(),
        # GaussianNB(),
        # DecisionTreeClassifier(),
        RandomForestClassifier(),
        # BaggingClassifier(),
        # MLPClassifier(max_iter=500)
    ]
    for mdl in mdls:
        # scores = sg_cross_val_score(
        #     mdl, X_train, y_col, 
        #     n_splits=5, n_jobs=-1, 
        #     scoring='f1_macro', groups=g_train
        # )
        score = latex_cross_val(mdl, X_train, y_train, g_train)
        cprint(f'{mdl.__class__.__name__}:\t{score:.2f}')
        # cprint (f'{np.mean(scores)}\t{mdl.__class__.__name__}\n\t{scores}\n', 'red')

# %%
