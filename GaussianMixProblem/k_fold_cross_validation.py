from globals import MseTestError
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import irls
import load_data
import mle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wls

def k_fold_cross_val(n_splits):
    
    """ Performs k fold cross validation and reports average test error.
    
        Parameters
        ----------
        n_splits : int
            number of splits to be taken. Usually taken to be 5.
        Returns
        -------
         MseTestError.mse_test_error : dict
            A nested dictionary that stores the average test error of the k 
            fold cross validation for each data set. 
    """
    
    kf = KFold(n_splits)
    for dataset in file_names:
        
        x = data[dataset]['x'].values
        y = data[dataset]['y'].values 
    
        test_mse_ols = 0.0
        test_mse_huber = 0.0 
        test_mse_tukey = 0.0 
        test_mse_mle = 0.0
        test_mse_wls = 0.0
    
        for train, test in kf.split(x):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            
            ols_coefs = smf.OLS(y_train, sm.add_constant(x_train)).fit().params
            
            irls_huber_coefs = irls.robust_irls_regression(x=x_train, 
                                                           y=y_train,
                                                           c=1.345,
                                                           penalty='Huber')
            irls_tukey_coefs = irls.robust_irls_regression(x=x_train, 
                                                           y=y_train,
                                                           c=4.685,
                                                           penalty='Tukey')
        
            mle_coefs = mle.mle_double_exp(x=x_train,
                                           y=y_train,
                                           alpha=np.array([0., 0.1]),
                                           num_steps=30000,
                                           learning_rate=1e-8)
        
            wls_coefs = wls.iterative_wls(x=x_train, y=y_train)
        
            y_pred_ols = ols_coefs[0] + ols_coefs[1]*x_test
            test_mse_ols += mean_squared_error(y_test, y_pred_ols)
        
            y_pred_huber = irls_huber_coefs[0] + irls_huber_coefs[1]*x_test
            test_mse_huber += mean_squared_error(y_test, y_pred_huber)
        
            y_pred_tukey = irls_tukey_coefs[0] + irls_tukey_coefs[1]*x_test
            test_mse_tukey += mean_squared_error(y_test, y_pred_tukey)
        
            y_pred_mle = mle_coefs[0] + mle_coefs[1]*x_test
            test_mse_mle += mean_squared_error(y_test, y_pred_mle)
        
            y_pred_wls = wls_coefs[0] + wls_coefs[1]*x_test
            test_mse_wls += mean_squared_error(y_test, y_pred_wls)
        
        
        avg_test_mse_ols = test_mse_ols/ n_splits
        MseTestError.mse_test_error[dataset]['OLS'] = avg_test_mse_ols
        
        avg_test_mse_huber = test_mse_huber/ n_splits
        MseTestError.mse_test_error[dataset]['Huber'] = avg_test_mse_huber
        
        avg_test_mse_tukey = test_mse_tukey/ n_splits
        MseTestError.mse_test_error[dataset]['Tukey'] = avg_test_mse_tukey
        
        avg_test_mse_mle = test_mse_mle/ n_splits
        MseTestError.mse_test_error[dataset]['MLE'] = avg_test_mse_mle
        
        avg_test_mse_wls = test_mse_wls/ n_splits
        MseTestError.mse_test_error[dataset]['WLS'] = avg_test_mse_wls
        
    

    return MseTestError.mse_test_error

if __name__ == '__main__':
    
    print 'Cross validation started...'
    
    data, file_names = load_data.load_data()
    import time
    t0 = time.time()
    test_error = k_fold_cross_val(n_splits=5)
    df = pd.DataFrame([(k, k1, v1) for k, v in test_error.items() 
                       for k1, v1 in v.items()], 
                      columns=['File_name', 'Method', 'Avg Test Error'])
    df.to_csv('avg_test_errors.csv', sep=',', index=False)
    t1 = time.time() # 5 fold Cross validation takes over 20 minutes to run.
    
    print 'Cross validation finished!'   
