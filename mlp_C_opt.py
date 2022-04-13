# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score, r2_score
from util_classes import sg_cross_val_score, reg_score, custom_cross_val_score_reg

from sklearn.neural_network import MLPClassifier


fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, length=180, overlap=120, 
    class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
)

# (X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
#     tables.WEEK, tables.METADATA, length=24000, overlap=18000, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )

# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from util_classes import custom_cross_val_score, latex_cross_val
mdl = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 20, 20))

space = [
    Categorical(['tanh', 'relu'], name='activation'),
    Real(1e-12, 1e+3, name='alpha', prior='log-uniform'),
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    return 1.0 - latex_cross_val(mdl, X_train, y_train, g_train, tabs=1)
    # scores = custom_cross_val_score(mdl, X_train, y_train, n_splits=5, n_jobs=-1, groups=g_train)
    # return 1.0 - np.mean(scores)
    # scores = custom_cross_val_score_reg(mdl, X_train, y_train, groups=g_train, n_jobs=-1)
    # return 1.0 - np.mean(scores)
    # return 1.0-np.mean(
    #     sg_cross_val_score(
    #         mdl, X_train, y_train, n_splits=10
    #         , n_jobs=-1, scoring='r2', groups=g_train
    #     )
    # )
from termcolor import cprint

res_gp = gp_minimize(objective, space, n_calls=10, verbose=True, n_jobs=-1)
cprint("\n\nBest score=%.4f" % (res_gp.fun), 'red')


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
print(res_gp.x)
# %% Optimal

# alpha = 1
alpha = res_gp.x[0]
# %%
# activation = res_gp.x[1]
# n_layers = res_gp.x[2]
# h_layers = tuple(res_gp.x[3:3+n_layers])
from termcolor import cprint


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

# mdl = MLPRegressor(max_iter=100000, hidden_layer_sizes=(100, 100, 20, 20), activation='tanh', alpha=alpha)
params = {'alpha': alpha}
mdl.set_params(**params)
print(mdl)

scores = custom_cross_val_score(
    mdl, X_train, y_train, 
    n_splits=5, n_jobs=-1, 
    groups=g_train
)
# scores = custom_cross_val_score_reg(mdl, X_train, y_train, groups=g_train, n_jobs=-1)
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)
# y_pred = y_pred_f.round().clip(0,3)
f1 = f1_score(y_test, y_pred, average='macro')
cprint(f'F1: {f1}', 'red')
print(classification_report(y_test, y_pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

from termcolor import cprint

# classes = np.unique(y_test)
# groups = np.unique(g_test)
# p_pred = []
# p_test = []
# for g in groups:
#     g_idx = np.where(g_test == g)
#     y_pred_g = y_pred_f[g_idx]
#     p_pred_g = np.mean(y_pred_g,0).round().clip(0,3)
#     p_pred += [p_pred_g]
#     p_test += [y_test[g_idx[0][0]]]

# print(classification_report(p_test, p_pred, zero_division=0))

# ConfusionMatrixDisplay.from_predictions(p_test, p_pred)

# %%
