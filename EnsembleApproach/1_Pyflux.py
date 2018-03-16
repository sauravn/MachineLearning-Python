import numpy as np
import pyflux as pf
#from pandas.io.data import DataReader
from pandas_datareader import data, wb
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

growthdata = pd.read_csv('http://www.pyflux.com/notebooks/GDPC1.csv')
USgrowth = pd.DataFrame(np.diff(np.log(growthdata['VALUE']))[149:len(growthdata['VALUE'])])
USgrowth.index = pd.to_datetime(growthdata['DATE'].values[1+149:len(growthdata)])
USgrowth.columns = ['US Real GDP Growth']
plt.figure(figsize=(15,5))
plt.plot(USgrowth)
plt.ylabel('Real GDP Growth')
plt.title('US Real GDP Growth')
plt.show()


model1 = pf.ARIMA(data=USgrowth, ar=4, ma=0)
model2 = pf.ARIMA(data=USgrowth, ar=8, ma=0)
model3 = pf.LLEV(data=USgrowth)
#model4 = pf.GASLLEV(data=USgrowth, family=pf.GASt())
model5 = pf.GPNARX(data=USgrowth, ar=1, kernel=pf.SquaredExponential())
model6 = pf.GPNARX(data=USgrowth, ar=2, kernel=pf.SquaredExponential())

mix = pf.Aggregate(learning_rate=1.0, loss_type='squared')
mix.add_model(model1)
mix.add_model(model2)
mix.add_model(model3)
#mix.add_model(model4)
mix.add_model(model5)
mix.add_model(model6)

print model6

mix.tune_learning_rate(40)
mix.learning_rate

mix.plot_weights(h=40, figsize=(15,5))
mix.show()
mix.summary(h=40)


'''
mix2 = pf.Aggregate(learning_rate=1.0, loss_type='squared')
mix2.add_model(model1)
mix2.add_model(model2)
mix2.add_model(model3)
mix2.add_model(model4)
mix2.tune_learning_rate(h=40)
mix2.learning_rate

mix2.plot_weights(h=40, figsize=(15,5))
mix2.summary(h=40)
'''