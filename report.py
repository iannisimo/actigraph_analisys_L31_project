# %%
import os
from turtle import title
from termcolor import colored, cprint
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables, fix_inline
from util_classes import sg_cross_val_score
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

# fix_inline()


(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=600, overlap=450, class_method='macs', 
    removeControl=False, scale='n', multiclass=False, version='7.3'
)
# %%
yX_train = np.zeros((y_train.shape[0], X_train.shape[1] + 1))
yX_train[:,0] = y_train
yX_train[:,1::] = X_train

df = pd.DataFrame(yX_train)

pr = ProfileReport(df, title='yX_train report', minimal=False)
pr.to_file('repp.html')
# %%
