# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from util_classes import sg_cross_val_score, custom_cross_val_score

from sklearn.neighbors import KNeighborsClassifier




# (X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
#     tables.AHA, tables.METADATA, length=180, overlap=120, 
#     class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
# )

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.WEEK, tables.METADATA, length=18000, overlap=12000, 
    class_method='macs', attrs='DND', scale='', split=True, test_pcentage=.25, max_zeros=.3, attrs_sbj='std', random_state=42
)

fix_inline()
# col = 1
# y_train = y_train[:,col]
# y_test = y_test[:,col]

# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from util_classes import latex_cross_val

mdl = KNeighborsClassifier(n_jobs=-1)

space = [
    Integer(2, 100, name='n_neighbors'),
    Categorical(['uniform', 'distance'], name='weights'),
    Categorical(['ball_tree', 'kd_tree', 'brute'], name='algorithm'),
    Integer(10, 200, name='leaf_size'),
    Integer(1, 10, name='p')
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    return 1.0 - latex_cross_val(mdl, X_train, y_train, g_train, tabs=1)
    # return 1.0 - np.mean(custom_cross_val_score(mdl, X_train, y_train, groups=g_train))
    # return 1.0-np.mean(sg_cross_val_score(
    #     mdl, X_train, y_train, 
    #     n_splits=5, n_jobs=-1, scoring='f1_macro', groups=g_train))

res_gp = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)
print("Best score=%.4f" % res_gp.fun)


plot_convergence(res_gp)
params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)
# %% Optimal



from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, classification_report

opt_cls = KNeighborsClassifier(n_jobs=-1, **params)
scores = sg_cross_val_score(opt_cls, X_train, y_train, n_splits=5, n_jobs=-1, groups=g_train)
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
fitted = opt_cls.fit(X_train, y_train)
print(f'Test score: {opt_cls.score(X_test, y_test)}')
y_pred = fitted.predict(X_test)
y_proba = fitted.predict_proba(X_test)
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(fitted, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%


from termcolor import cprint
from sklearn.metrics import ConfusionMatrixDisplay

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
