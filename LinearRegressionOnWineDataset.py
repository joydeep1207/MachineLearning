# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 12:41:25 2016

@author: Joydeep1207
"""

# Dataset from 
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn. cross_validation import cross_val_score
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('winequality-red.csv', sep=';')

#The Dataframe.describe() method calculates summary statistics
#for each column of the dataframe
print (df.describe())

# Visualizing the data can help in indicate relationships exist between 
# the response variable and the explanatory variables
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Vs Quality')
plt.show()

# A weak positive relationship between the alcohol content and quality is visible in the
# scatter plot in the preceding figure; wines that have high alcohol content are often
# high in quality

# similarly for 
plt.scatter(df['volatile acidity'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Volatile acidity')
plt.title('Volatile acidity Vs Quality')
plt.show()

# we get another relationship of quality with volatile acidity. He we have 
# linear regression with multiple response variable


# How can we decide which explanatory variables to include in the model?
#datafeame.corr() pairwise correlation matrix
df.corr()


# Lets go and get our hands dirty
df = pd.read_csv('winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']

# train_test_split function to randomly partition the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print 'R-squared:', regressor.score(X_test, y_test)

# lets get cross valiidation score 
scores = cross_val_score(regressor, X, y, cv=10)
print scores.mean(), scores

# plot with true quality and predicted quality
plt.scatter( y_predictions , y_test)
plt.xlabel('Predicted quality')
plt.ylabel('True quality')
plt.title('Predicted quality Vs  True Quality')
plt.show()

# fitting models using gradient decent 
# SGDRegressor is an implementation of SGD that can be used even for
# regression problems with more features. It can be used
# to optimize different cost functions to fit different linear models. By default, it will
# optimize the residual sum of squares

# we will use Boston Housing data set
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data,data.target)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print 'Cross validation r-squared scores:', scores
print 'Average cross validation r-squared score:', np.mean(scores)

regressor.fit_transform(X_train, y_train)
print 'Test set r-squared score', regressor.score(X_test, y_test)


## --------------End of script ------------------------



