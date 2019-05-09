#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mglearn
import numpy as np
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsRegressor
# instantiate the model, set the number of neighbors to consider to 5
reg = KNeighborsRegressor(n_neighbors=5)
# Fit the model using the Training set and Training targets
reg.fit(X_train, y_train)

KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=3, p=2,
                     weights='uniform')

# Predictions on the Test set
reg.predict(X_test)

# Evaluate the model
reg.score(X_test, y_test)


# Visualizing the five nearest neighbors
# mglearn.plots.plot_knn_regression(n_neighbors=5)

# Showing predictions for all possible features values

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Create 1000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
plt.suptitle("Nearest Neighbor Regression")
for n_neighbors, ax in zip([1, 5, 15], axes):
        # Make predictions using 1, 5 or 15 neighbors
        reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
        ax.plot(X, y, 'o')
        ax.plot(X, -3 * np.ones(len(X)), 'o')
        ax.plot(line, reg.predict(line))
        ax.set_title("%d neighbor(s)" % n_neighbors)