'''
Problem Description:
Please fit a model using the data provided and send your predicted values back to me by February 7:
 
Using some set of the variables a-h in the file intern_data.csv, create a model to predict y
Use your model to create predicted values of y given the variables a-h in the file intern_test.csv
Respond with a file intern_predicted.csv which contains two columns:
i: the index from intern_test.csv
y: your predicted values
Please also include the code you used to generate the predicted values in your response. 
There are no restrictions on programming language.
'''

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv('intern_data.csv')
test = pd.read_csv('intern_test.csv')


#factors that will predict the y
desired_factors = ['a','b','c','d','e','f','g','h']

#Conver column c and h to category codes
train['c'] = train['c'].astype('category').cat.codes
train['h'] = train['h'].astype('category').cat.codes
test['c'] = test['c'].astype('category').cat.codes
test['h'] = test['h'].astype('category').cat.codes

#set my model to DecisionTree
model = linear_model.LinearRegression()

#set prediction data to factors that will predict, and set target to y
train_data = train[desired_factors]
test_data = test[desired_factors]
target = train.y
#target_missing = test.y

#fitting model with prediction data and telling it my target
model.fit(train_data, target)

predictions = model.predict(test_data)
#print predictions
#print test_data
y = pd.DataFrame(np.array(predictions), columns = list("y"))
print y
i = test.iloc[:,0]
print pd.concat([i, y], axis=1)
with open('intern_predicted.csv', 'w') as f:
	pd.concat([i, y], axis=1).to_csv(f)
#print "Score:", model.score(target_missing, predictions)
