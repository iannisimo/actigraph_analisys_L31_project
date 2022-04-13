# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score
from util_classes import sg_cross_val_score, reg_score, custom_cross_val_score_reg

from sklearn.svm import SVR

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, length=120, overlap=-60, 
    class_method='macs', attrs='DND', scale='', split=True, max_zeros=.3
)
# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

mdl = SVR(max_iter=1000000)

# sw = np.ones(y_train.shape)
# sw[y_train == 0] = .5
# sw[y_train == 1] = .7
# sw[y_train == 2] = .3
# sw[y_train == 3] = 5.
sw = None

space = [
    Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    Real(1e-6, 1e+30, name='C', prior='log-uniform'),
    Real(1e-6, 1e+6, name='gamma', prior='log-uniform'),
    Integer(1, 8, name='degree', prior='uniform'),
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    scores = sg_cross_val_score(
        mdl, X_train, y_train, 
        n_splits=5, 
        groups=g_train, n_jobs=-1,
        scoring='max_error'
    )
    return -np.median(scores)
    # scores = custom_cross_val_score_reg(mdl, X_train, y_train, groups=g_train, n_jobs=-1, sample_weight=sw)
    # s = np.mean(scores)
    # return 1.0 - s

res_gp = gp_minimize(objective, space, n_calls=15, verbose=True, n_jobs=-1)
print("\n\nBest score=%.4f" % (res_gp.fun))


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
# %% Optimal

params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from termcolor import cprint

opt_cls = SVR(max_iter=10000000, **params)
scores = sg_cross_val_score(
    opt_cls, X_train, y_train, 
    n_splits=5, n_jobs=-1, 
    groups=g_train, 
    scoring=reg_score
)
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
opt_cls.fit(X_train, y_train, sample_weight=sw)
cprint(f'Train score: {opt_cls.score(X_train, y_train)}', 'red')
cprint(f'Test score: {opt_cls.score(X_test, y_test)}', 'red')
y_pred_f = opt_cls.predict(X_test)
y_pred = y_pred_f.round().clip(0,3)
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

ConfusionMatrixDisplay.from_predictions(p_test, p_pred)

# %%
