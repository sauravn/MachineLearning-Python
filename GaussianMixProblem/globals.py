import glob
from collections import defaultdict

''' A class to store coefficients of linear regression as a nested dict'''

class Coefs(object):
    estimated_coefs = defaultdict(dict) # global data
    for file_names in glob.glob('*.csv'):
        estimated_coefs[file_names]['OLS'] = []  
        estimated_coefs[file_names]['Huber'] = []
        estimated_coefs[file_names]['Tukey'] = []
        estimated_coefs[file_names]['WLS'] = []
        estimated_coefs[file_names]['MLE'] = []
        
''' A class to store MSE test errors as a nested dict'''
class MseTestError(object):
    mse_test_error = defaultdict(dict) # global data
    for file_names in glob.glob('*.csv'):
        mse_test_error[file_names]['OLS'] = 0.0  
        mse_test_error[file_names]['Huber'] = 0.0
        mse_test_error[file_names]['Tukey'] = 0.0
        mse_test_error[file_names]['WLS'] = 0.0
        mse_test_error[file_names]['MLE'] = 0.0