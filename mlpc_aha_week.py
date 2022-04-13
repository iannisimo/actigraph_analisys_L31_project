# %% Load reqirements

import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from util_classes import subject_pred, subject_pred_prob
from latex_utils import latex_confusion_matrix as lcm

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier

fix_inline()

# %% Load data


(X_train, y_train, g_train, X_test_a, y_test_a, g_test_a, scaler) = load_dataset(
    tables.AHA, tables.METADATA, length=120, overlap=60, 
    class_method='macs', attrs='t', scale='',
    test_pcentage=.25
)

(X_w, y_w, g_w, _, _, _, _) = load_dataset(
    tables.WEEK, tables.METADATA, length=28800, overlap=28800, 
    class_method='macs', attrs='t', scale='',
    split=False, max_zeros=.3
)

MIN = 0
MAX = 3

# %% Train the model

# mdl = MLPRegressor(hidden_layer_sizes=(100, 40, 20), activation='tanh', alpha=0.005101621000226295)
mdl = MLPClassifier(hidden_layer_sizes=(100, 20, 20), activation='tanh', alpha=0.005101621000226295)

mdl.fit(X_train, y_train)

# %% Predict training values from AHA

y_ptrain_f = mdl.predict(X_train)
y_ptrain = y_ptrain_f.round().clip(MIN, MAX)
lcm(y_train, y_ptrain, 'Finestre')

y_ptrain_p = mdl.predict_proba(X_train)
p_train, p_ptrain = subject_pred_prob(y_train, y_ptrain_p, g_train)

lcm(p_train, p_ptrain, 'Soggetti')

# %% Predict test values from AHA

y_pred_a_f = mdl.predict(X_test_a)
y_pred_a = y_pred_a_f.round().clip(MIN, MAX)

lcm(y_test_a, y_pred_a, 'Finestre')

y_pred_a_p = mdl.predict_proba(X_test_a)
p_test_a, p_pred_a = subject_pred_prob(y_test_a, y_pred_a_p, g_test_a)

lcm(p_test_a, p_pred_a, 'Soggetti')

# %% Predict all values from WEEK

y_pred_W_f = mdl.predict(X_w)
y_pred_W = y_pred_W_f.round().clip(MIN, MAX)

lcm(y_w, y_pred_W, 'Finestre')

y_pred_W_p = mdl.predict_proba(X_w)
p_test_W, p_pred_W = subject_pred_prob(y_w, y_pred_W_p, g_w)

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

y_pred_tw_p = mdl.predict_proba(X_test_w)
p_test_w, p_pred_tw = subject_pred_prob(y_test_w, y_pred_tw_p, g_test_w)

lcm(p_test_w, p_pred_tw, 'Soggetti')

# %%
