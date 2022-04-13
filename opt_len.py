# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import classification_report
from util_classes import sg_cross_val_score
from termcolor import cprint

from sklearn.ensemble import RandomForestClassifier

fix_inline()
# %%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

mdl = RandomForestClassifier(n_jobs=-1)
logo = LeaveOneGroupOut()

space = [
    Integer(10, 900, name='length', prior='log-uniform'),
    Integer(10, 900, name='overlap', prior='log-uniform'),
]

@use_named_args(space)

def objective(**params):
    cprint(f'Params: {params}', 'green', attrs=['bold'])
    (X, y, g, _, _, _) = load_dataset(
        tables.AHA, tables.METADATA, 
        class_method='aha3', 
        removeControl=True, scaled=True, multiclass=False,
        **params
    )

    return 1.0 - np.mean(cross_val_score(mdl, X, y, groups=g, scoring='f1_weighted', n_jobs=-1, cv=logo))

res_gp = gp_minimize(objective, space, n_calls=20, verbose=True, n_jobs=-1)
print("\n\nBest score=%.4f" % (1.0 - res_gp.fun))


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
# %% Optimal

params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(
    tables.AHA, tables.METADATA, 
    class_method='aha3', 
    removeControl=True, scaled=True, multiclass=False,
    **params
)

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix



scores = cross_val_score(
    mdl, X_train, y_train, 
    cv=logo, n_jobs=-1, 
    groups=g_train, scoring='f1_weighted')
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
fitted = mdl.fit(X_train, y_train)
print(f'Test score: {mdl.score(X_test, y_test)}')
predicted = fitted.predict(X_test)
print(classification_report(y_test, predicted))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(fitted, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%
