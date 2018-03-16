from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv('intern_data.csv', index_col=0)
feature_cols = ['a','b','c','d','e','f','g','h']
data['c'] = data['c'].astype('category').cat.codes
data['h'] = data['h'].astype('category').cat.codes

X = data[feature_cols]
y = data.y 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
lm = LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()
print "Score:", model.score(X_test, y_test)
print "mae:", mean_absolute_error(y_test, predictions)
print "r2_score:", r2_score(y_test, predictions)