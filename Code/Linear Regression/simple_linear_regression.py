#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("breslow.csv")
X = dataset.iloc[:, 5].values
y = dataset.iloc[:, 1].values

X = X.reshape(-1, 1)
# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# Different from y_test
# y_test is the real data observed
# y_pred is the predicted data from the Linear Regression model
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('The number of smoker vs Age (Training Set)')
plt.xlabel('The number of smoker')
plt.ylabel('Age')
plt.show

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('The number of smoker vs Age (Test Set)')
plt.xlabel('The number of smoker')
plt.ylabel('Age')
plt.show




