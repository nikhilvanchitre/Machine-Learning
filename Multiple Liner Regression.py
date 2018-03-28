# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:09:04 2018

@author: nikhi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
dir = input("Enter Directory")
os.chdir(dir)
# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable trap
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple linear model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Backward elimination(automated)
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
sl = 0.05
X_opt = X[:,[0,1,2,3,4,5]]
X_model = backwardElimination(X_opt, sl)
