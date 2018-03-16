import urllib2
import StringIO
import csv
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import pyflux as pf
from datetime import datetime
from pandas_datareader import data, wb

 
startdate = '20111118'
enddate   = '20121125'
 
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
Wn = 0.06 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')
 
# Second, apply the filter
tempf = signal.filtfilt(B,A, temp)
 
# Make plots
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(date,temp, 'b-')
plt.plot(date,tempf, 'r-',linewidth=2)
plt.xlabel("Timestamp")
plt.ylabel("CPU Usage (%)")
plt.legend(['Original','Filtered'])
plt.title("CPU usage after passing it through Bandpass Filter")
ax1.axes.get_xaxis().set_visible(False)
 
ax1 = fig.add_subplot(212)
plt.plot(date,temp-tempf, 'b-')
plt.ylabel("CPU Usage (%)")
plt.xlabel("Timestamp")
plt.legend(['Residuals'])
#plt.show()

# Filtered data
#print cpu_data
fig2 = plt.figure(figsize=(15,5))
plt.plot(date,tempf, 'r-',linewidth=2)
plt.xlabel("Timestamp")
plt.ylabel("Filtered CPU Usage (%)")
plt.legend(['Filtered'])
plt.title("Filtered CPU usage")
plt.show()

cpu_data = tempf
print "start ensemble"
model1 = pf.ARIMA(data=cpu_data, ar=4, ma=0)
model2 = pf.ARIMA(data=cpu_data, ar=8, ma=0)
model3 = pf.LLEV(data=cpu_data)
#model4 = pf.GASLLEV(data=cpu_data, family=pf.GASt())
model4 = pf.GASLLEV(data=cpu_data, family=pf.Poisson())
model5 = pf.GPNARX(data=cpu_data, ar=1, kernel=pf.SquaredExponential())
model6 = pf.GPNARX(data=cpu_data, ar=2, kernel=pf.SquaredExponential())
model7 = pf.DynReg('CPUusage', data=dataframe)

mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')
mix.add_model(model1)
mix.add_model(model2)
mix.add_model(model3)
#mix.add_model(model4)
#mix.add_model(model5)
#mix.add_model(model6)
mix.add_model(model7)


mix.tune_learning_rate(20)
print mix.learning_rate

mix.plot_weights(h=20, figsize=(15,5))
#plt.show()

print mix.summary(h=20)
print mix.predict_is(h=5)
