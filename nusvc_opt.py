# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from util_classes import sg_cross_val_score
from termcolor import cprint

from sklearn.svm import NuSVC

from utils import fix_inline
fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=100, overlap=50, class_method='macs', 
    removeControl=False, scale='n', multiclass=True
)

col = 1
y_train = y_train[:,col]
y_test = y_test[:,col]

# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective, plot_regret, plot_evaluations

mdl = NuSVC(max_iter=100000)

space = [
    Real(1e-6, 1e-1, name='nu', prior='log-uniform'),
    Real(1e-6, 1e2, name='gamma', prior='log-uniform'),
    Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    Integer(1, 5 , name='degree', prior='log-uniform'),
    # Categorical(['scale', 'auto'], name='gamma')
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    scores = sg_cross_val_score(
        mdl, X_train, y_train, 
        n_splits=5, scoring='f1_macro', 
        groups=g_train, n_jobs=-1
    )
    s = 1.0 - np.mean(scores)
    v = np.var(scores)
    return s

res_gp = gp_minimize(objective, space, n_calls=50, verbose=True, n_jobs=-1)
print("\n\nBest score=%.4f" % (1.0 - res_gp.fun))


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
plot_regret(res_gp)
plot_evaluations(res_gp)
# %% Optimal

params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

opt_cls = NuSVC(max_iter=100000, **params)
scores = sg_cross_val_score(
    opt_cls, X_train, y_train, 
    n_splits=5, n_jobs=-1, 
    groups=g_train, scoring='f1_macro')
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
fitted = opt_cls.fit(X_train, y_train)
print(f'Test score: {opt_cls.score(X_test, y_test)}')
predicted = fitted.predict(X_test)
print(classification_report(y_test, predicted))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(fitted, X_test, y_test, ax=ax)
# %%
