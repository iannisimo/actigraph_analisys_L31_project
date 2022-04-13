# %% Load reqirements

import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from util_classes import subject_pred
from latex_utils import latex_confusion_matrix as lcm
from termcolor import cprint

from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestClassifier

fix_inline()

# %% Load data

(X_train, y_train, g_train, X_test_a, y_test_a, g_test_a, _) = load_dataset(
    tables.WEEK, tables.METADATA, length=14400, overlap=-5760, 
    class_method='m5', attrs='DND', scale='',
    test_pcentage=.25
)

(X_w, y_w, g_w, _, _, _, _) = load_dataset(
    tables.WEEK, tables.METADATA, length=14400, overlap=2400, 
    class_method='m5', attrs='DND', scale='',
    split=False, max_zeros=.3
)


MIN = 0
MAX = 3

# sw = np.ones(y_train.shape)
# sw[y_train == 0] = 2
# sw[y_train == 1] = 3
# sw[y_train == 2] = .05
# sw[y_train == 3] = 100

# scaler_aha = StandardScaler()
# scaler_aha.fit(X_train)
# X_train = scaler_aha.transform(X_train)
# X_test_a = scaler_aha.transform(X_test_a)

# %% Train the model

mdl = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=20, n_estimators=25)
# mdl = NuSVR()
# mdl = SVR(verbose=True, kernel='rbf', C=1.7856780170631506e+16, gamma=0.0071126376839912376, degree=2)
# mdl = SVR(verbose=True, tol=1e-9)
# 
mdl.fit(X_train, y_train)

# %% Predict training values from AHA

y_ptrain_f = mdl.predict(X_train)
y_ptrain = y_ptrain_f.round().clip(MIN, MAX)

lcm(y_train, y_ptrain, 'Finestre')

p_train, p_ptrain = subject_pred(y_train, y_ptrain_f, g_train)

lcm(p_train, p_ptrain, 'Soggetti')

# %% Predict test values from AHA

y_pred_a_f = mdl.predict(X_test_a)
y_pred_a = y_pred_a_f.round().clip(MIN, MAX)

lcm(y_test_a, y_pred_a, 'Finestre')

p_test_a, p_pred_a = subject_pred(y_test_a, y_pred_a_f, g_test_a)

lcm(p_test_a, p_pred_a, 'Soggetti')

# %% Predict all values from WEEK

y_pred_W_f = mdl.predict(X_w)
y_pred_W = y_pred_W_f.round().clip(MIN, MAX)

lcm(y_w, y_pred_W, 'Finestre')

p_test_W, p_pred_W = subject_pred(y_w, y_pred_W_f, g_w)

lcm(p_test_W, p_pred_W, 'Soggetti')


# %% Predict test values from WEEK

X_test_w = X_w[np.where(np.logical_not(np.in1d(g_w, g_train)))]
y_test_w = y_w[np.where(np.logical_not(np.in1d(g_w, g_train)))]
g_test_w = g_w[np.where(np.logical_not(np.in1d(g_w, g_train)))]

X_train_w = X_w[np.where(np.in1d(g_w, g_train))]
y_train_w = y_w[np.where(np.in1d(g_w, g_train))]
g_train_w = g_w[np.where(np.in1d(g_w, g_train))]

# X_test_w = X_w[np.where(np.in1d(g_w, g_test_a))]
# y_test_w = y_w[np.where(np.in1d(g_w, g_test_a))]
# g_test_w = g_w[np.where(np.in1d(g_w, g_test_a))]

# scaler_week = StandardScaler()
# scaler_week.fit(X_train_w)
# X_train_w = scaler_week.transform(X_train)
# X_test_w = scaler_week.transform(X_test_w)

y_pred_tw_f = mdl.predict(X_test_w)
y_pred_tw = y_pred_tw_f.round().clip(MIN, MAX)

lcm(y_test_w, y_pred_tw, 'Finestre')

p_test_w, p_pred_tw = subject_pred(y_test_w, y_pred_tw_f, g_test_w)

lcm(p_test_w, p_pred_tw, 'Soggetti')

# %%
