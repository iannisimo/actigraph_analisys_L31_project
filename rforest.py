# %%

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", message='.*')


import os
from termcolor import colored, cprint
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables, fix_inline
from util_classes import sg_cross_val_score
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix, f1_score
from matplotlib import pyplot as plt

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=100, overlap=100, class_method='aha4', 
    removeControl=False, scaled=True, multiclass=True
)
# y_train = y_train[:,2]
# X_train = normalize(X_train)
# pca = PCA(n_components=5).fit(X_train)
# X_train = pca.transform(X_train)

# %%
# mdl = RandomForestClassifier(n_estimators=200)
# mdl.fit(X_train, y_train)
# y_train_pred = mdl.predict(X_train)
# y_test_pred = mdl.predict(X_test)
# tr_score = f1_score(y_train, y_train_pred)
# print(f'Train score:\t{tr_score}')
# plot_confusion_matrix(mdl, X_train, y_train)
# te_score = f1_score(y_test, y_test_pred)
# print(f'Test score:\t{te_score}')
# plot_confusion_matrix(mdl, X_test, y_test)


# %%
for col in range(0,y_train.shape[1]):
    cprint(f'\n\n\tCurrent Column\t{col}\n\n', 'green')
    y_train_col = y_train[:,col]
    y_test_col = y_test[:,col]
    mdl = RandomForestClassifier(n_estimators=200)
    mdl.fit(X_train, y_train_col)
    y_train_pred = mdl.predict(X_train)
    y_test_pred = mdl.predict(X_test)
    tr_score = f1_score(y_train_col, y_train_pred)
    print(f'Train score:\t{tr_score}')
    plot_confusion_matrix(mdl, X_train, y_train_col, display_labels=[f'X_train_{col}', f'y_train_{col}'])
    te_score = f1_score(y_test_col, y_test_pred)
    print(f'Test score:\t{te_score}')
    plot_confusion_matrix(mdl, X_test, y_test_col, display_labels=[f'X_test_{col}', f'y_test_{col}'])

# %%
