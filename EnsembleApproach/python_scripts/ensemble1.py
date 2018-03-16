# AdaBoost Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
url = "/home/hadoop/bitbrains_data/fastStorage/2013-8/pre_data1.csv"
names = ['Timestamp', 'CPU_usage', 'Memory_capacity', 'Memory_usage']
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
dataframe[['CPU_usage','Memory_usage']] = dataframe[['CPU_usage','Memory_usage']].apply(pandas.to_numeric)
#print(dataframe.index)
y=list(range(len(dataframe.index)))
print(dataframe.CPU_usage[:5])
dataframe.CPU_usage[:10000].plot(x='CPU_usage', y=y)
plt.show()

'''
array = dataframe.values
X = array[:,0:4]
Y = array[:,4]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
'''
