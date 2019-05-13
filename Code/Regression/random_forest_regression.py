#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("breslow.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

# Fitting the Decision Tree Regression to the dataset 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[45]])

# Visualising Random Forest Regression results
# Get more precise results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Death at a certain age (Random Forest Regression)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()
