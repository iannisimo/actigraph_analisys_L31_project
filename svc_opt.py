# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score
from util_classes import sg_cross_val_score, custom_cross_val_score
from util_classes import latex_cross_val

from sklearn.svm import SVC

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, length=180, overlap=120, 
    class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
)

# def subject_score(y_true, y_proba, *, groups):
#     classes = np.unique(y_true)
#     u_groups = np.unique(groups)
#     p_pred = []
#     p_test = []
#     for g in u_groups:
#         g_idx = np.where(g_test == g)
#         y_proba_g = y_proba[g_idx]
#         sum_proba = np.median(y_proba_g,0)
#         p_class_idx = np.unravel_index(sum_proba.argmax(), sum_proba.shape)[0]
#         p_pred += [classes[p_class_idx]]
#         p_test += [y_true[g_idx[0][0]]]
#     return f1_score(p_test, p_pred, average='weighted')


# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

mdl = SVC(max_iter=1000000, probability=True)

space = [
    Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    Real(1e-6, 1e+6, name='C', prior='log-uniform'),
    Real(1e-6, 1e+1, name='gamma', prior='log-uniform'),
    Integer(1, 8, name='degree', prior='log-uniform'),
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    return 1.0 - latex_cross_val(mdl, X_train, y_train, g_train, tabs=1)
    # scores = sg_cross_val_score(
    #     mdl, X_train, y_train, 
    #     n_splits=5, scoring='f1_weighted', 
    #     groups=g_train, n_jobs=-1
    # )
    # scores = custom_cross_val_score(mdl, X_train, y_train, groups=g_train)
    # s = 1.0 - np.mean(scores)
    return s

res_gp = gp_minimize(objective, space, n_calls=25, verbose=True, n_jobs=-1)
print("\n\nBest score=%.4f" % (1.0 - res_gp.fun))


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)
# %% Optimal



from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

opt_cls = SVC(max_iter=10000000, probability=True, **params)
scores = sg_cross_val_score(
    opt_cls, X_train, y_train, 
    n_splits=5, n_jobs=-1, 
    groups=g_train, scoring='f1_weighted')
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
opt_cls.fit(X_train, y_train)
print(f'Test score: {opt_cls.score(X_test, y_test)}')
y_pred = opt_cls.predict(X_test)
y_proba = opt_cls.predict_proba(X_test)
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(opt_cls, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
# %%
from termcolor import cprint

classes = np.unique(y_test)
groups = np.unique(g_test)
p_pred = []
p_test = []
for g in groups:
    g_idx = np.where(g_test == g)
    y_proba_g = y_proba[g_idx]
    sum_proba = np.median(y_proba_g,0)
    p_class_idx = np.unravel_index(sum_proba.argmax(), sum_proba.shape)[0]
    p_pred += [classes[p_class_idx]]
    p_test += [y_test[g_idx[0][0]]]

ConfusionMatrixDisplay.from_predictions(p_test, p_pred)
# %%
# %%
