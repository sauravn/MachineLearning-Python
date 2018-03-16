

import urllib2
import StringIO
import csv
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
 
startdate = '20111118'
enddate   = '20121125'
 
#url = "/home/saurav/Downloads/bitbrains_data/fastStorage/2013-8/tmp.csv"
url = "/home/saurav/Downloads/bitbrains_data/rnd/2013-8/1.csv"
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
series = Series.from_csv('/home/saurav/Downloads/bitbrains_data/fastStorage/2013-8/1_corr_Diskread.csv', header=0, index_col=1)

autocorrelation_plot(series)
plt.show()

plot_acf(series, lags=31)
plt.show()