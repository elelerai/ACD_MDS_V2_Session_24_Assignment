# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:03:40 2019

@author: Eliud Lelerai
"""
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
import pydotplus
from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.metrics import classification_report

Url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"

titanic = pd.read_csv(Url)
titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic.head()
titanic.info()
titanic.
# select features
y = titanic['Survived']
X = titanic[['Pclass','Sex','Age','SibSp','Parch','Fare']]

# categorical data

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
X['Sex'] = class_le.fit_transform(X['Sex'].values)

# Missing Values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(X)
X = imr.transform(X.values)

       
       
# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = model_selection.train_test_split (X, y, test_size=0.3, random_state=0)

# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)
score = dtree.score(X_test, y_test)
print(score)
      
# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)
print("The predictions are:",y_pred)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

