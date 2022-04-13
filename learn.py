
# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from operator import le
from utils import load_dataset, tables
import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression, PoissonRegressor, TweedieRegressor, SGDRegressor

from sklearn.linear_model import RidgeCV, SGDClassifier, Perceptron
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
(X_train, y_train, _, _) = load_dataset(tables.AHA, tables.METADATA, length=240, overlap=60, class_method='aha2')

# %%

from skopt.space import Real, Integer, Categorical 
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

n_features = X_train.shape[1]

mdl = GradientBoostingRegressor(n_estimators=50, random_state=0)

# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

@use_named_args(space)

def objective(**params):

    mdl.set_params(**params)

    return -np.mean(cross_val_score(mdl, X_train, y_train, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%d
- min_samples_split=%d
- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                            res_gp.x[2], res_gp.x[3],
                            res_gp.x[4]))

plot_convergence(res_gp)
print('hello')
# %%
# - max_depth=2
# - learning_rate=0.097698
# - max_features=71
# - min_samples_split=17
# - min_samples_leaf=30
nmdl = GradientBoostingRegressor(n_estimators=50, random_state=0, max_depth=2, learning_rate=0.097698, max_features=71, min_samples_split=17, min_samples_leaf=30)

scores = cross_val_score(nmdl, X_train, y_train, cv=5, n_jobs=-1)
print(scores)
# mdls = [
#     RandomForestRegressor(),
#     BaggingRegressor(),
#     GradientBoostingRegressor(),
#     KNeighborsRegressor(),
#     LinearRegression(),
#     PoissonRegressor(),
#     TweedieRegressor(),
#     # SGDRegressor()
#     ]
# %%
mdls = [
    RidgeCV(),
    SGDClassifier(),
    Perceptron(),
    LinearSVC(),
    NuSVC(),
    SVC(),
    KNeighborsClassifier(),
    GaussianProcessClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    BaggingClassifier(),
    MLPClassifier(solver='adam', hidden_layer_sizes=(100,))
]
results = dict()
for mdl in mdls:
    scores = cross_val_score(mdl, X_train, y_train, cv=5, n_jobs=-1)
    results[mdl.__class__.__name__] = scores
    print (f'{np.mean(scores)}\t{mdl.__class__.__name__}\n\t{scores}\n')
for r in results:
    print(r, results[r])
# %%
