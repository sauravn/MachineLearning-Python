import numpy as np
import pyflux as pf
#from pandas.io.data import DataReader
from pandas_datareader import data, wb
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


url = "/home/saurav/Downloads/bitbrains_data/fastStorage/2013-8/tmp.csv"
#names = ['Timestamp', 'CPU_usage', 'Memory_capacity', 'Memory_usage']
names = ['Timestamp', 'CPUcores', 'CPUcapacity', 'CPUusageMHZ', 'CPUusage', 'Memorycapacity', 'Memoryusage', 'Diskread', 'Diskwrite', 'NetworkReceived', 'NetworkTransmitted']
#print(dataframe.index)
dataframe = pd.read_csv(url, names=names)

dataframe['CPUusage'] = pd.to_numeric(dataframe['CPUusage'], errors='coerce')
dataframe['Diskwrite'] = pd.to_numeric(dataframe['Diskwrite'], errors='coerce')
dataframe['Diskread'] = pd.to_numeric(dataframe['Diskread'], errors='coerce')
dataframe['Memoryusage'] = pd.to_numeric(dataframe['Memoryusage'], errors='coerce')
dataframe['Memorycapacity'] = pd.to_numeric(dataframe['Memorycapacity'], errors='coerce')
y = list(range(len(dataframe.index)))

array = dataframe.values
X = array[1:,5:6]
y = array[1:,6]

'''
dataframe[['CPU_usage','Memory_usage']] = dataframe[['CPU_usage','Memory_usage']].apply(pd.to_numeric)
#print(dataframe.index)
y=list(range(len(dataframe.index)))
#cpu_data=dataframe[['CPU_usage']]
cpu_data=dataframe.CPU_usage[:1000]
print cpu_data
dataframe.CPU_usage[:1000].plot(x='CPU_usage', y=y)
'''
dataframe['CPUusage'].round(4)
y=list(range(len(dataframe.index)))
#cpu_data=dataframe[['CPU_usage']]
cpu_data=dataframe.CPUusage[1:]
#print cpu_data
plt.figure(figsize=(15,5))
dataframe.CPUusage[1:].plot(x='CPUusage', y=y)
plt.ylabel('CPU usgae')
plt.title('Times series analysis of CPU usage')
plt.show()

print "start ensemble"
model1 = pf.ARIMA(data=cpu_data, ar=4, ma=0)
model2 = pf.ARIMA(data=cpu_data, ar=8, ma=0)
model3 = pf.LLEV(data=cpu_data)
#model4 = pf.GASLLEV(data=cpu_data, family=pf.GASt())
model4 = pf.GASLLEV(data=cpu_data, family=pf.Poisson())
model5 = pf.GPNARX(data=cpu_data, ar=1, kernel=pf.SquaredExponential())
model6 = pf.GPNARX(data=cpu_data, ar=2, kernel=pf.SquaredExponential())

mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')
mix.add_model(model1)
#mix.add_model(model2)
mix.add_model(model3)
mix.add_model(model4)
#mix.add_model(model5)
#mix.add_model(model6)


mix.tune_learning_rate(4)
print mix.learning_rate

mix.plot_weights(h=40, figsize=(15,5))
#plt.show()

print mix.summary(h=40)
print mix.predict_is(h=5)
