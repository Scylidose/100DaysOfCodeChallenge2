#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("breslow.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[47]]))))

# Visualising SVR results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Death at a certain age (SVR)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()

# Visualising SVR results
# Get more precise results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Death at a certain age (Polynomial Regressor)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()
