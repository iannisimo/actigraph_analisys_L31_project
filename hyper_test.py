# %%

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

from utils import fix_inline
fix_inline()
from util_classes import fix_pgf
fix_pgf()

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Create the dataset
rng = np.random.RandomState(42)
X = np.linspace(0, 3, 7)[:, np.newaxis]
X_t = np.linspace(1/4, 11/4, 6)[:, np.newaxis]
# X_t = np.setdiff1d(np.linspace(0, 3, 19)[:, np.newaxis], X)
XX = np.linspace(0, 3, 1000)[:, np.newaxis]
# y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
# y = X**2 + 2*X * np.log(X)
y = (X ** 3 - 5 * X ** 2 + 6.5 * X - 1) ** 2
y = np.array(y)[:,0] + rng.normal(0, 0.1, X.shape[0])
yy = (XX ** 3 - 5 * XX ** 2 + 6.5 * XX - 1) ** 2
y_t = (X_t ** 3 - 5 * X_t ** 2 + 6.5 * X_t - 1) ** 2

regr_1 = Pipeline(steps=[
    ('preprocessor', PolynomialFeatures(degree=3, include_bias=False)),
    ('estimator', Ridge(alpha=1e-9))
])

regr_2 = Pipeline(steps=[
    ('preprocessor', PolynomialFeatures(degree=6, include_bias=False)),
    ('estimator', Ridge(alpha=1e-9))
])

regr_3 = Pipeline(steps=[
    ('preprocessor', PolynomialFeatures(degree=9, include_bias=False)),
    ('estimator', Ridge(alpha=1e-9))
])

# yy = np.sin(XX).ravel() + np.sin(6 * XX).ravel()

# Fit regression model
# regr_1 = RandomForestRegressor()
# regr_1 = KNeighborsRegressor(n_neighbors=3, leaf_size=30, weights='distance')
# regr_1 = MLPRegressor(hidden_layer_sizes=(20, 20, 20), alpha=1e-5, activation='tanh', max_iter=1000, tol=1e-9)
# regr_1 = SVR(degree=8, kernel='poly', gamma='scale', epsilon=.05, coef0=3, C=1)
# regr_1 = SVR(C=1e+2, degree=10, epsilon=1e-3)
# regr_1 = NuSVR(nu=.7, C=1e+20, gamma='auto', kernel='rbf')


# regr_2 = AdaBoostRegressor(
#     DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
# ) 

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# Predict
y_1 = regr_1.predict(XX)
y_2 = regr_2.predict(XX)
y_3 = regr_3.predict(XX)
# y_2 = regr_2.predict(X)

# Plot the results

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10, 8))

# ax1.scatter(X, y, c="k", label="Punti di training")
# ax1.plot(XX, yy,  c="m", label='Funzione reale')
# ax1.legend(loc='lower right')
# ax2.scatter(X, y, c="k", label="Punti di training")
# ax2.plot(XX, y_1, c="g", label="Grado 5", linewidth=1)
# ax2.legend(loc='lower right')
# ax3.scatter(X, y, c="k", label="Punti di training")
# ax3.plot(XX, y_2, c="r", label="Grado 7", linewidth=1)
# ax3.legend(loc='lower right')
# ax4.scatter(X, y, c="k", label="Punti di training")
# ax4.plot(XX, y_3, c="b", label="Grado 9", linewidth=1)
# ax4.legend(loc='lower right')

# plt.savefig('plots/reg_hpar.pgf')

from sklearn.metrics import mean_squared_error

plt.figure()
plt.scatter(X_t, y_t, c="k")
plt.plot(XX, yy,  c="r", label='Funzione reale', linewidth=1)
plt.scatter(X, y, c="k", label='Sample + rumore')
plt.ylim(-3,3)
plt.legend(loc='lower left')
plt.show()
plt.savefig('plots/scatter.pgf')

plt.figure()
plt.scatter(X, y, c="k", label="Punti per training")
plt.scatter(X_t, y_t, c="r", label="Punti per validazione")
plt.plot(XX, yy,  c="r", label='Funzione reale', linewidth=1)
plt.plot(XX, y_1, c="g", label="Grado 3", linewidth=1)
plt.plot(XX, y_2, c="m", label="Grado 6", linewidth=1)
plt.plot(XX, y_3, c="b", label="Grado 9", linewidth=1)
plt.ylim(-3,3)
plt.legend(loc='lower left')
plt.show()
plt.savefig('plots/reg_hpar.pgf')

y_1t = regr_1.predict(X_t)
y_2t = regr_2.predict(X_t)
y_3t = regr_3.predict(X_t)
y_1T = regr_1.predict(X)
y_2T = regr_2.predict(X)
y_3T = regr_3.predict(X)

print(mean_squared_error(y_t, y_1t))
print(mean_squared_error(y_t, y_2t))
print(mean_squared_error(y_t, y_3t))

print(mean_squared_error(y, y_1T))
print(mean_squared_error(y, y_2T))
print(mean_squared_error(y, y_3T))
# %%

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA


# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()


# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
n_neighbors = 3

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T, y_, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()