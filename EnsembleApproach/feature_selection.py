
# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
# load data
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
X = array[1:,5:9]
Y = array[1:,4]
print X
print Y

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_