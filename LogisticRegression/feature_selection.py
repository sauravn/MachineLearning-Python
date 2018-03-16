
# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
# load data
url = "intern_data.csv"

names = ['a','b','c','d','e','f','g','h','y']
dataframe = read_csv(url, names=names)

dataframe['c'] = dataframe['c'].astype('category').cat.codes
dataframe['h'] = dataframe['h'].astype('category').cat.codes
dataframe['y'] = pd.to_numeric(dataframe['y'], errors='coerce')

dataframe1 = dataframe[['a','b','c','d','e','f','g','h','y']]
array = dataframe1.values
X = array[1:,0:8]
Y = array[1:,8]

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y.astype(int))
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_