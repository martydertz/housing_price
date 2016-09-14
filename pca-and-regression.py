# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C:\kaggle\house prices")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.shape

# 81 variables - ahh, the curse of dimensionality
from sklearn import linear_model, decomposition
from sklearn.preprocessing import scale
pca = decomposition.PCA()
features=list(train.columns).remove('SalePrice')
target = train['SalePrice']
def get_numeric_cols(df):
    objectCols =[]
    numericCols = []
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            numericCols.append(col)
        else:
            objectCols.append(col)
    return(df[numericCols])
numeric_df = get_numeric_cols(train)

numeric_df = numeric_df.fillna(numeric_df.mean())
numeric_df = scale(numeric_df)
pca.fit(numeric_df)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
# Amount of variance explained for each component
var = pca.explained_variance_ratio_
# Sum of explained variance
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
pca.n_components = 15
numeric_reduced = pca.fit_transform(numeric_df)
numeric_reduced.shape
regr = linear_model.LinearRegression()
regr.fit(numeric_reduced, target)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(numeric_reduced) - target) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(numeric_reduced, target))
test = scale(get_numeric_cols(test.fillna(test.mean())))
pca.n_components = 15
test_reduced = pca.fit_transform(test)
predictions = regr.predict(test_reduced)
test = pd.read_csv("test.csv")
test['SalePrice'] = predictions
submission = test[['Id', 'SalePrice']]
submission[['Id','SalePrice']].to_csv("predictions.csv", index = False)
