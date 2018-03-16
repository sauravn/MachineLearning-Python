from sklearn import datasets
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset from scikit's data sets
data = pd.read_csv('intern_data.csv', index_col=0)
feature_cols = ['a','b','c','d','e','f','g','h']
data['c'] = data['c'].astype('category').cat.codes
data['h'] = data['h'].astype('category').cat.codes

#X = data[feature_cols]
#Y = data.y 


dataframe1 = data[['a','b','c','d','e','f','g','h','y']]
array = dataframe1.values
#X = array[0:,0:8]
X = data.e.values
#X = array[1:,6]
#Y = array[1:,8]
Y = data.y.values
#Y = Y.reshape(-1,1)
print X
print Y

#X, y = diabetes.data, diabetes.target

# Instanciate a GP model
gp = GaussianProcess(regr='constant', corr='absolute_exponential',
                     theta0=[1e-4] * 10, thetaL=[1e-12] * 10,
                     thetaU=[1e-2] * 10, nugget=1e-2, optimizer='Welch')

# Fit the GP model to the data performing maximum likelihood estimation
gp.fit(X, Y)

# Deactivate maximum likelihood estimation for the cross-validation loop
gp.theta0 = gp.theta_  # Given correlation parameter = MLE
gp.thetaL, gp.thetaU = None, None  # None bounds deactivate MLE

# Perform a cross-validation estimate of the coefficient of determination using
# the cross_validation module using all CPUs available on the machine
K = 20  # folds
R2 = cross_val_score(gp, X, y=Y, cv=KFold(Y.size, K), n_jobs=1).mean()
print("The %d-Folds estimate of the coefficient of determination is R2 = %s"
      % (K, R2))