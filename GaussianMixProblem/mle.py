import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def neg_log_lhood_double_exp(x, y, beta, alpha):

    #--------------------------------------------------------------------
    # Compute the negative log likelihood for the double exponential aka
    # the cost function to be minimized.
    #--------------------------------------------------------------------
    cost_function = 0
    n = len(x)

    for i in range(n):

        log_var_term = 0.5*(alpha[0] + alpha[1] * x[i])
        inverse_std_dev_term = np.exp(-0.5*(alpha[0] + alpha[1]*x[i]))
        l1_term = np.abs(y[i] - beta[0] - beta[1]*x[i])
        cost_function += log_var_term + inverse_std_dev_term*l1_term

    return (cost_function + 0.0)/ n

def update_double_exp_wrt_beta(x, y, beta, alpha, learning_rate):

    beta_0_grad, beta_1_grad = 0, 0
    n = len(x)

    #---------------------------------------------------
    # Compute gradient of cost function with respect to
    # beta = (beta_0, beta_1)
    #---------------------------------------------------
    for i in range(n):

        inverse_std_dev_term = np.exp(-0.5*(alpha[0] + alpha[1]*x[i]))
        sgn_term = np.sign(y[i] - beta[0] - beta[1]*x[i])
        beta_0_grad += sgn_term*inverse_std_dev_term
        beta_1_grad += sgn_term*x[i]*inverse_std_dev_term

    beta_0_grad = -((beta_0_grad + 0.0)/ n)
    beta_1_grad = -((beta_1_grad + 0.0)/ n)

    #----------------------------------------------
    # Update beta and return the updated value
    #----------------------------------------------
    beta_0 = beta[0] - learning_rate * beta_0_grad
    beta_1 = beta[1] - learning_rate * beta_1_grad
    return np.array([beta_0, beta_1])


def update_double_exp_wrt_alpha(x, y, beta, alpha, learning_rate):

    alpha_0_grad, alpha_1_grad = 0, 0
    n = len(x)

    #---------------------------------------------------
    # Compute gradient of cost function with respect to
    # alpha = (alpha_0, alpha_1)
    #---------------------------------------------------
    for i in range(n):

        inverse_std_dev_term = np.exp(-0.5*(alpha[0] + alpha[1]*x[i]))
        l1_term = np.abs(y[i] - beta[0] - beta[1]*x[i])
        alpha_0_grad += inverse_std_dev_term*l1_term
        alpha_1_grad += x[i] * (1 - inverse_std_dev_term*l1_term)

    alpha_0_grad = 0.5 - ((alpha_0_grad)/ 2*n)
    alpha_1_grad = ((alpha_1_grad + 0.0)/ 2*n)

    #----------------------------------------------
    # Update alpha and return the updated value
    #----------------------------------------------
    alpha_0 = alpha[0] - learning_rate*alpha_0_grad
    alpha_1 = alpha[1] - learning_rate*alpha_1_grad
    return np.array([alpha_0, alpha_1])

def mle_double_exp(x, y, alpha, num_steps, learning_rate):

    """Run a maximum likelihood linear regression with l1 penalty and
       a pre-chosen function for variance.
        Parameters
        ----------
        x : float
            predictor vector of size n*1
        y : float
            target variable of size n*1
        alpha : numpy array of size 2*1
                initial estimates of alpha
        num_steps : int
                    Maximum number of steps for MLE
        learning_rate : float
                       Learning rate for gradient descent
        Returns
        -------
        coefs : float
                2*1 vector of coefficients of linear regression.
        Helper functions
        ----------------
        neg_log_likelihood_double_exp(x, y, beta, alpha)
            Computes the cost function
        update_double_exp_wrt_beta(x, y, beta, alpha, learning_rate)
            Updates beta by moving in the direction of negative beta_gradient
        update_double_exp_wrt_alpha(x, y, beta, alpha, learning_rate)
            Updates alpha by moving in the direction of negative alpha_gradient
        """

    cost_function_vtr = np.zeros(num_steps)
    beta = smf.OLS(y, sm.add_constant(x)).fit().params # Take OLS estimates as 
                                                       # initial estimates.
    for step in range(num_steps):

        #-----------------------
        # Compute cost function
        #-----------------------
        cost_function_vtr[step] = neg_log_lhood_double_exp(x, y, beta, alpha)

        #-------------------
        # Update parameters
        #-------------------
        beta = update_double_exp_wrt_beta(x, y, beta, alpha, learning_rate)
        alpha = update_double_exp_wrt_alpha(x, y, beta, alpha, learning_rate)

    #------------------------------------------------------
    # Make a plot of objective function vs number of steps
    #------------------------------------------------------
    #plt.plot(list(range(num_steps)), cost_function_vtr)
    #plt.show()

    return beta