import numpy as np 
import numpy.linalg as LA
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.robust.scale as sm_scale

def huber_weight(z, c):
    
    #--------------------------------------------
    # Returns weight for Huber penalty
    #--------------------------------------------
    if abs(z) <= c:
        return 1.0
    else:
        return c/(0.0 + abs(z))
huber_weight = np.vectorize(huber_weight)


def tukey_weight(z, c):
    
    #--------------------------------------------
    # Returns weight for Tukey's biweight penalty
    #--------------------------------------------
    if abs(z) <= c:
        return (1.0 - (z/c)**2)**2
    else:
        return 0
tukey_weight = np.vectorize(tukey_weight)

def robust_irls_regression(x, y, c, penalty, max_iter=100, tol=1e-8):
    
    """Run a robust linear regression with iterated weighted least squares
        Parameters
        ----------
        x : float
            predictor vector of size n*1
        y : float
            target variable of size n*1
        
        penalty : string
            Type of penalty to be applied while doing the iterated weighted
            least squares (IWLS) regression. Defualts to OLS regression.
            Choices for penalty are 'Huber', 'Tukey'. 
        
        c: float
            Tuning hyperparameter depending on the chosen penalty.
            Defaults to none.
            
        max_iter : int
            Maximum number of iterations for IWLS. Default is 100
        
        tol: float
            tolerance level for norm of difference of successive estimates for
            coefficients of linear regression and robust estimates of spread
            of residuals. Defaults to 1e-8
    
        Returns
        -------
        coefs : float
            2*1 vector of coefficients of linear regression. 
        """
    x = np.c_[np.ones(len(x)), x] # append column vector of 1's
    
   #------------------------------------------ 
    # Initial co-efficients, returned by OLS.
   #------------------------------------------
    results = smf.WLS(y, sm.add_constant(x)).fit()
    coefs = results.params
    
    #--------------------------------------------
    # Raise error if no penalty is specified
    #----------------------------------------
    if penalty is None:
        raise ValueError("Specify either 'Huber' or 'Tukey' penalty!")
    
    old_coefs = coefs
    residuals = results.resid
    
    #--------------------------------------------
    # Initial estimate for spread of residuals
    #--------------------------------------------
    robust_sd = sm_scale.mad(residuals)
    old_robust_sd = robust_sd
    
    #--------------------------------------------
    # Initialize weights
    #--------------------------------------------
    weights = np.diag(1.0/(residuals**2))
    
    for iteration in range(max_iter):
        
        #-------------------------------
        # Update regression coefficients 
        #-------------------------------
        coefs = LA.solve(np.dot(np.dot(x.T, weights), x), 
                         np.dot(np.dot(x.T, weights), y))
        
        #--------------------------------------------
        # Update residuals
        #--------------------------------------------
        residuals = y - np.dot(coefs, x.T)
        
        #--------------------------------------------
        # Update robust measure of spread of residuals 
        #--------------------------------------------
        robust_sd = sm_scale.mad(residuals)
        
        #--------------------------------------------
        # Standardize updated residuals
        #--------------------------------------------
        standardized_residuals = residuals/ robust_sd
        
        #--------------------------------------------
        # Update weights
        #--------------------------------------------
        if penalty == 'Huber':
            weights = np.diag(huber_weight(standardized_residuals, c=c))
        elif penalty == "Tukey":
            weights = np.diag(tukey_weight(standardized_residuals, c=c))
        
        #--------------------------------------------
        # Stop if estimates for co-efficients and spread of residuals 
        # are stable.
        #--------------------------------------------
        if LA.norm(robust_sd - old_robust_sd) < tol and \
           LA.norm(coefs - old_coefs) < tol:
            break
            
        old_coefs = coefs
        old_robust_sd = robust_sd
        
    return coefs