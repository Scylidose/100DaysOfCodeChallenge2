#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("breslow.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

# Fitting the Decision Tree Regression to the dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Visualising Decision Tree Regression results
# Get more precise results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Death at a certain age (Decisional Tree Regressor)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()
