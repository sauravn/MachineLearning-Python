import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.nonparametric.kernel_regression as nparam_kreg

def iterative_wls(x, y, tol=1e-6, max_iter=100):
    
    """Run a weighted least squares linear regression with
       iterative refinement of variance. (This is computationally intensive!)
        Parameters
        ----------
        x : float
            predictor vector of size n*1
        y : float
            target variable of size n*1
        max_iter : int
                   Maximum number of iterations for IWLS. Default is 100
        tol: float
             tolerance level for norm of difference of successive estimates for
             coefficients of linear regression and robust estimates of spread
             of residuals. Defaults to 1e-6
        Returns
        -------
        coefs : float
            2*1 vector of coefficients of linear regression.
        """
        
    x = np.c_[np.ones(len(x)), x] # append column vector of 1's
    iteration = 0
    old_coefs = None

    #----------------------------------------
    # Run an OLS to get initial estimates
    #----------------------------------------
    regression = smf.WLS(y, sm.add_constant(x)).fit()
    coefs = regression.params

    while old_coefs is None or (np.max(abs(coefs - old_coefs)) > tol and
                                iteration < max_iter): 
    
        #----------------------------------------------------------------------
        # Construct the log-squared residuals and use a non-parametric
        # method (kernel regression) to estimate the conditonal mean. 
        # Residual can be 0 in which case log-squared residual is not defined.
        # Ignore the warning and put a small value for log-squared residual and
        # proceed. 
        
        # Exponentiate to predict the variance and take inverse of the variance 
        # as weights.
        #----------------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'): 

            old_coefs = coefs
            log_squared_residuals = np.where(regression.resid**2 > 0, 
                                         np.log(regression.resid**2), 
                                         1e-12)
            model = nparam_kreg.KernelReg(endog=y,
                                      exog=log_squared_residuals,
                                      var_type='c')
            weights = np.exp(model.fit()[0])**-1

            #-------------------------------
            # Update regression coefficients
            #-------------------------------
            regression = sm.WLS(y, sm.add_constant(x), weights=weights).fit()
            coefs = regression.params
            iteration += 1

    return coefs