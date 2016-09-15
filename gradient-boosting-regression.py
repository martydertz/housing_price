# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:25:03 2016

@author: martDawg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("C:\kaggle\house prices")
train = pd.read_csv("train.csv", index_col = 'Id')
test = pd.read_csv("test.csv", index_col = 'Id')
# Plotting Sales prices and log-sales prices
train['SalePrice'].plot.hist()
log_sale_price = np.log(train['SalePrice'])
log_sale_price.plot.hist()
train_features = train.drop('SalePrice', axis = 1)
train_features = train_features.select_dtypes(include=[np.number]) 
classVars = ['MSSubClass','OverallQual','OverallCond', 
                              'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
                              'MoSold', 'YrSold']   
train_features = train_features.drop(classVars, axis = 1)   
train_features = train_features.fillna(train_features.mean())
train_features = train_features.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis = 1)                     
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
offset = int(train.shape[0] * 0.9)
X_train, y_train = train_features[:offset], log_sale_price[:offset]
X_test, y_test = train_features[offset:], log_sale_price[offset:]

###############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(log_sale_price, clf.predict(train_features))
print("MSE: %.4f" % mse)

###############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)
    
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, train_features.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
##############################################################################
# Predicting for submission'
test = pd.read_csv("test.csv", index_col='Id')
test_features= test.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis = 1)
test_features = test_features.select_dtypes(include=[np.number]) 
test_features = test_features.drop(classVars, axis = 1)   
test_features = test_features.fillna(test_features.mean())
test_features.shape, train_features.shape
predictions = clf.predict(test_features)
test = pd.read_csv("test.csv")
test['SalePrice'] = np.exp(predictions)
submission = test[['Id', 'SalePrice']]
submission[['Id','SalePrice']].to_csv("predictions_bosting.csv", index = False)
