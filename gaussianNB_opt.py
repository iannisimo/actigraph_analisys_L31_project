# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import classification_report
from util_classes import sg_cross_val_score

from sklearn.naive_bayes import GaussianNB

fix_inline()

(X_train, y_train, g_train, X_test, y_test, g_test, _) = load_dataset(
    tables.AHA, tables.METADATA, 
    length=100, overlap=-1, class_method='macs', 
    removeControl=False, scale='n'
)

# %%
# col = 1
# y_train = y_train[:,col]
# y_test = y_test[:,col]
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

mdl = GaussianNB()

space = [
    Real(1e-12, 1e2, name='var_smoothing', prior='log-uniform'),
]

@use_named_args(space)

def objective(**params):
    mdl.set_params(**params)
    return 1.0-np.mean(sg_cross_val_score(mdl, X_train, y_train, n_splits=5, n_jobs=-1, scoring='accuracy', groups=g_train))

res_gp = gp_minimize(objective, space, n_calls=50, random_state=42)
print("\n\nBest score=%.4f" % (1.0 - res_gp.fun))


plot_convergence(res_gp)
plot_objective(res_gp, n_points = 10)
# %% Optimal

params = {}
for i, x in enumerate(res_gp.x):
    params[space[i].name] = x

print(params)


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

opt_cls = GaussianNB(**params)
scores = sg_cross_val_score(opt_cls, X_train, y_train, n_splits=5, n_jobs=-1, groups=g_train)
print(f'Scores: {scores}\n\tmean: {np.mean(scores)}')
fitted = opt_cls.fit(X_train, y_train)
predicted = fitted.predict(X_test)
print(classification_report(y_test, predicted))

fig, ax = plt.subplots(1,1)
plot_confusion_matrix(fitted, X_test, y_test, ax=ax)
# sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%
