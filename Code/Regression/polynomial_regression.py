#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("breslow.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 2)
X_poly = polynomial.fit_transform(X)
linear2 = LinearRegression()
linear2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear.predict(X), color='blue')
plt.title("Death at a certain age (Linear Regressor)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()

# Visualising the Polynomial Regression results
# Get more precise results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear2.predict(polynomial.fit_transform(X_grid)), color='blue')
plt.title("Death at a certain age (Polynomial Regressor)")
plt.xlabel("Age")
plt.ylabel("Number of death (smoke)")
plt.show()

# Predicting a new result with Linear Regression
linear.predict([[47]])

# Predicting a new result with Polynomial Regression
linear2.predict(polynomial.fit_transform([[47]]))

