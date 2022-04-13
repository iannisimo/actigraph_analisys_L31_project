# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score
from util_classes import sg_cross_val_score, custom_cross_val_score

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=480, overlap=120, class_method='macs', 
    removeControl=False, scale='s', multiclass=False,
)


# %%
mdl = SVR(verbose=1)

mdl.fit(X_train, y_train)
y_pred_f = mdl.predict(X_test)
y_pred = y_pred_f.round().clip(0,3)

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

from termcolor import cprint

classes = np.unique(y_test)
groups = np.unique(g_test)
p_pred = []
p_test = []
for g in groups:
    g_idx = np.where(g_test == g)
    y_pred_g = y_pred_f[g_idx]
    p_pred_g = np.mean(y_pred_g,0).round().clip(0,3)
    p_pred += [p_pred_g]
    p_test += [y_test[g_idx[0][0]]]

print(classification_report(p_test, p_pred))

ConfusionMatrixDisplay.from_predictions(p_test, p_pred)
# %%