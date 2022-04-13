# %%
import os
os.chdir('/home/simone/Tesi/python/scratch/')
from utils import load_dataset, tables, fix_inline
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor

(X_train, y_train, g_train, X_test, y_test, g_test) = load_dataset(tables.AHA, tables.METADATA, length=120, overlap=60, class_method='aha')

fix_inline()

# %%
from skopt.space import Real, Integer, Categorical 
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

reg = KNeighborsRegressor(n_jobs=-1)

space = [
    Integer(2, 100, name='n_neighbors'),
    Categorical(['uniform', 'distance'], name='weights'),
    Categorical(['ball_tree', 'kd_tree', 'brute'], name='algorithm'),
    Integer(10, 200, name='leaf_size'),
    Integer(1, 4, name='p')
]

@use_named_args(space)

def objective(**params):
    reg.set_params(**params)
    return -np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1, groups=g_train, scoring='neg_mean_absolute_error'))

res_gp = gp_minimize(objective, space, n_calls=50, random_state=42)
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
# %% Optimal

# res_gp.x = [2, 'distance', 'ball_tree', 36, 4]
from matplotlib import pyplot as plt
import seaborn as sns

opt_reg = KNeighborsRegressor(n_neighbors=4, weights='distance', algorithm='ball_tree', leaf_size=200, p=1, n_jobs=-1)
scores = cross_val_score(opt_reg, X_train, y_train, cv=5, n_jobs=-1, groups=g_train)
print(f'Scores: {scores}')
fitted = opt_reg.fit(X_train, y_train)
predicted = fitted.predict(X_test)

fig, ax = plt.subplots(1,1)
sns.scatterplot(x=predicted, y=y_test, ax=ax)
 # %%
