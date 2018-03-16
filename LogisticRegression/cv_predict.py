import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv('intern_data.csv', index_col=0)
feature_cols = ['a','b','c','d','e','f','g','h']
data['c'] = data['c'].astype('category').cat.codes
data['h'] = data['h'].astype('category').cat.codes

X = data[feature_cols]
Y = data.y 

print X
print Y

lm = LinearRegression()
scores = cross_val_score(lm, X, Y, cv=10, scoring='mean_squared_error')
print(scores)
# fix the sign of MSE scores
mse_scores = -scores
print(mse_scores)
# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)
# calculate the average RMSE
print(rmse_scores.mean())