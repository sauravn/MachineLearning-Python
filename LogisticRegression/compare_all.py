# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
'''
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
'''
url = "intern_data.csv"

names = ['a','b','c','d','e','f','g','h','y']
dataframe = pd.read_csv(url, names=names)

dataframe['c'] = dataframe['c'].astype('category').cat.codes
dataframe['h'] = dataframe['h'].astype('category').cat.codes
dataframe['y'] = pd.to_numeric(dataframe['y'], errors='coerce')

dataframe1 = dataframe[['a','b','c','d','e','f','g','h','y']]
array = dataframe1.values
X = array[1:,0:8]
y = array[1:,8]

# feature extraction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape
lm = LogisticRegression()
model = lm.fit(X_train, y_train.astype(int))
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()
print "Score:", model.score(X_test, y_test)
print "mae:", mean_absolute_error(y_test, predictions)
print "r2_score:", r2_score(y_test, predictions)
'''
# prepare configuration for cross validation test harness
seed = 10
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''