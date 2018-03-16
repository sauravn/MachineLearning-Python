import numpy as np
import pyflux as pf
#from pandas.io.data import DataReader
from pandas_datareader import data, wb
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


url = "/home/saurav/Downloads/bitbrains_data/fastStorage/2013-8/pre_data1.csv"
names = ['Timestamp', 'CPU_usage', 'Memory_capacity', 'Memory_usage']
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
usageData = pd.read_csv(url, names=names)
cpu_data = pd.DataFrame(np.diff(np.log(usageData['CPU_usage']))[1:len(usageData['CPU_usage'])])
#print cpu_data

'''
dataframe[['CPU_usage','Memory_usage']] = dataframe[['CPU_usage','Memory_usage']].apply(pd.to_numeric)
#print(dataframe.index)
y=list(range(len(dataframe.index)))
#cpu_data=dataframe[['CPU_usage']]
cpu_data=dataframe.CPU_usage[:1000]
print cpu_data
dataframe.CPU_usage[:1000].plot(x='CPU_usage', y=y)
'''
print cpu_data
plt.plot(cpu_data)
plt.ylabel('CPU usgae')
plt.title('Times series analysis of CPU usage')
#plt.show()

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
