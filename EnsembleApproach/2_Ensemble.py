import numpy as np
import pyflux as pf
#from pandas.io.data import DataReader
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from pandas_datareader import data, wb
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def filter(dataframe, Y):
	# Break the file into two lists
	date, temp = [],[]
	date = list(range(len(dataframe.index)-1))
	temp = Y
	#print date
	#print temp
	 
	# temp needs to be converted from a "list" into a numpy array...
	temp = np.array(temp)
	temp = temp.astype(np.float) #...of floats

	# First, design the Buterworth filter
	N  = 2    # Filter order
	Wn = 0.04 # Cutoff frequency
	B, A = signal.butter(N, Wn, output='ba')
	 
	# Second, apply the filter
	tempf = signal.filtfilt(B,A, temp)
	return tempf

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
X = array[1:,6:7]
Y = array[1:,4]
print X
print Y
#X.astype(int)
#Y.astype(int)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
'''
plt.plot(cpu_data)
plt.ylabel('CPU usgae')
plt.title('Times series analysis of CPU usage')
'''
