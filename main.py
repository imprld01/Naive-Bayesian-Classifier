# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:09:25 2017

@author: bowns
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

gnb = GaussianNB()
bcd = datasets.load_breast_cancer()

data = bcd.data
target = bcd.target

result = np.zeros(target.shape, target.dtype)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)

for k, (train_index, test_index) in enumerate(skfold.split(data, target)):
    print("Fold:", k+1)
    print("Train Index:", train_index)
    print("Test Index:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_class, test_class = target[train_index], target[test_index]
    
    gnb.fit(train_data, train_class)
    train_predict = gnb.predict(train_data)
    test_predict = gnb.predict(test_data)
    
    print(train_class)
    print(train_predict)
    print(test_class)
    print(test_predict)
    
    result[test_index] = test_predict
    score = (test_predict==test_class).sum()/test_index.size
    
    print("Accuracy:", score)
    print()

gnb_scores = cross_val_score(gnb, data, target, scoring='accuracy', cv=skfold)
print(gnb_scores)