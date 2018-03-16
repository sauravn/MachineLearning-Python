from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import pandas as pd

url = "/home/saurav/Downloads/bitbrains_data/fastStorage/2013-8/tmp.csv"
#names = ['Timestamp', 'CPU_usage', 'Memory_capacity', 'Memory_usage']
names = ['Timestamp', 'CPUcores', 'CPUcapacity', 'CPUusageMHZ', 'CPUusage', 'Memorycapacity', 'Memoryusage', 'Diskread', 'Diskwrite', 'NetworkReceived', 'NetworkTransmitted']
#print(dataframe.index)
dataframe = pd.read_csv(url, names=names)
dataframe.round(4)
dataframe['Diskwrite'] = pd.to_numeric(dataframe['Diskwrite'], errors='coerce')
dataframe['Diskread'] = pd.to_numeric(dataframe['Diskread'], errors='coerce')
dataframe['Memoryusage'] = pd.to_numeric(dataframe['Memoryusage'], errors='coerce')
dataframe['Memorycapacity'] = pd.to_numeric(dataframe['Memorycapacity'], errors='coerce')
array = dataframe.values
X = array[1:,5:6]
y = array[1:,6]
print X
print y

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