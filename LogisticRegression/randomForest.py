# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import utils

import pdb

data = pd.read_csv('intern_data.csv')
test = pd.read_csv('intern_test.csv')

#factors that will predict the y
feature_cols = ['a','b','c','d','e','f','g','h']
data['c'] = data['c'].astype('category').cat.codes
data['h'] = data['h'].astype('category').cat.codes

X = data[feature_cols]
y = data.y 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

lm = RandomForestClassifier()
X_train = X_train.astype('int')
y_train = y_train.astype('int')
X_test = X_test.astype('int')
y_test = y_test.astype('int')
print(utils.multiclass.type_of_target(X_train))
print(utils.multiclass.type_of_target(y_train))
model = lm.fit(X_train, y_train)
print lm
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()
print "Score:", model.score(X_test, y_test)
print "mae:", mean_absolute_error(y_test, predictions)
print "r2_score:", r2_score(y_test, predictions)
print "Train Accuracy :: ", accuracy_score(y_train, model.predict(X_train))
print "Test Accuracy  :: ", accuracy_score(y_test, predictions)