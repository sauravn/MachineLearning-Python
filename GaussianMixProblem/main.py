from globals import Coefs
import irls
import matplotlib.pyplot as plt
import load_data
import mle
import numpy as np
import pandas as pd
import aggregate_data
import statsmodels.formula.api as smf
import statsmodels.api as sm
import wls

#------------------------------------------------------------------------------
def run_ols():
    
    # Run a ordinary least squares regression
    
    print 'Finding OLS estimates ...'
    
    for dataset in file_names:
        
        predictor = data[dataset]['x'].values
        response = data[dataset]['y'].values 
        ols_coefs = smf.OLS(response, sm.add_constant(predictor)).fit().params
        Coefs.estimated_coefs[dataset]['OLS'].append(ols_coefs)
    
    print 'Finished finding OLS estimates!'
        
def run_irls_huber(tuning_parameter):
    
    # Run a robust huber Regression with a pre-chosen tuning hyper-parameter.
    # The co-efficients are found via a iterated re-weighted least squares
    # approach
    
    print 'Finding IRLS Huber estimates ...'
    
    for dataset in file_names:
        
        predictor = data[dataset]['x'].values
        response = data[dataset]['y'].values 
        huber_reg_coefs = irls.robust_irls_regression(x=predictor, 
                                                      y=response,
                                                      c=tuning_parameter,
                                                      penalty='Huber') 
        Coefs.estimated_coefs[dataset]['Huber'].append(huber_reg_coefs)
        
    print 'Finished finding IRLS Huber estimates!'
    
def run_irls_tukey(tuning_parameter):
    
    # Run a robust Tukey Regression with a pre-chosen tuning hyper-parameter.
    # The co-efficients are found via a iterated re-weighted least squares
    # approach
    
    print 'Finding IRLS Tukey estimates ...'
    
    for dataset in file_names:
        
        predictor = data[dataset]['x'].values
        response = data[dataset]['y'].values 
        tukey_reg_coefs = irls.robust_irls_regression(x=predictor, 
                                                      y=response,
                                                      c=tuning_parameter,
                                                      penalty='Tukey') 
        Coefs.estimated_coefs[dataset]['Tukey'].append(tukey_reg_coefs)
    
    print 'Finished finding IRLS Tukey estimates!'

def run_wls():
    
    # Run a weighted least squares regression with an iterative refinement of 
    # variance  regression to tackle heteroskedasticity.
    
    print 'Finding weighted least squares estimates ...'
    
    for dataset in file_names:
        
        predictor = data[dataset]['x'].values
        response = data[dataset]['y'].values 
        wls_coefs = wls.iterative_wls(x=predictor, y=response)
        Coefs.estimated_coefs[dataset]['WLS'].append(wls_coefs)
        
    print 'Finished weighted least squares estimates!'

def run_mle_double_exp(num_steps, alpha, learning_rate):  
    
    # Run a robust maximum likelihood formulation using gradient descent.
    # Take initial coeffs as OLS coeffs and pre-specify initial alpha and
    # learning rate
    
    print 'Finding robust MLE estimates...'
   
    for dataset in file_names:
        
        predictor = data[dataset]['x'].values
        response = data[dataset]['y'].values 
        mle_coefs = mle.mle_double_exp(x=predictor, 
                                       y=response, 
                                       alpha=alpha,
                                       num_steps=num_steps,
                                       learning_rate=learning_rate)
        Coefs.estimated_coefs[dataset]['MLE'].append(mle_coefs)
    
    print 'Finished finding robust MLE estimates!'   
#------------------------------------------------------------------------------   

#----------------------------------------------------------
# Plot the fitted line for every appoach and each data set.
#---------------------------------------------------------
def plot():

    plt.style.use('ggplot')

    count = 0
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 20)

    for dataset in file_names:

        row, col = count / 3, count % 3
        x, y = data[dataset]['x'], data[dataset]['y']
        
        # Range of x-values
        x_new = np.array([data[dataset]['x'].min(), data[dataset]['x'].max()])
        
        # Scatter plot of data
        axs[row, col].scatter(x, y, color='g')

        # OLS fit
        beta_0_ols = Coefs.estimated_coefs[dataset]['OLS'][0][0]
        beta_1_ols = Coefs.estimated_coefs[dataset]['OLS'][0][1]
        ols, = axs[row, col].plot(x_new, 
                                  beta_0_ols + beta_1_ols*x_new,
                                  c='red', 
                                  linewidth=2, 
                                  label='OLS')
        
        # IRLS Huber fit
        beta_0_huber = Coefs.estimated_coefs[dataset]['Huber'][0][0]
        beta_1_huber = Coefs.estimated_coefs[dataset]['Huber'][0][1]
        huber, = axs[row, col].plot(x_new, 
                                    beta_0_huber + beta_1_huber*x_new,
                                    c='blue', 
                                    linewidth=2, 
                                    label='Huber')
        
        # IRLS Tukey fit
        beta_0_tukey = Coefs.estimated_coefs[dataset]['Tukey'][0][0]
        beta_1_tukey = Coefs.estimated_coefs[dataset]['Tukey'][0][1]
        tukey, = axs[row, col].plot(x_new, 
                                    beta_0_tukey + beta_1_tukey*x_new,
                                    c='cyan', 
                                    linewidth=2, 
                                    label='Tukey')
        
        # WLS fit
        beta_0_wls = Coefs.estimated_coefs[dataset]['WLS'][0][0]
        beta_1_wls = Coefs.estimated_coefs[dataset]['WLS'][0][1]
        wls, = axs[row, col].plot(x_new, 
                                  beta_0_wls + beta_1_wls*x_new,
                                  c='magenta', 
                                  linewidth=2, 
                                  label='WLS')
        
        # MLE fit 
        beta_0_mle = Coefs.estimated_coefs[dataset]['MLE'][0][0]
        beta_1_mle = Coefs.estimated_coefs[dataset]['MLE'][0][1]
        mle_double_exp, = axs[row, col].plot(x_new, 
                                             beta_0_mle + beta_1_mle*x_new,
                                             c='black', 
                                             linewidth=2, 
                                             label='MLE_double_exp')

        # Labels
        axs[row, col].legend(handles=[ols, huber, tukey, wls, mle_double_exp],
                             fontsize=8, framealpha=0.7)
        axs[row, col].set_xlabel('x')
        axs[row, col].set_ylabel('y')
        axs[row, col].set_title(dataset, fontsize=14)

        count += 1
    
    fig.delaxes(axs[-1, -1]) 
    fig.savefig('linear_fits.pdf')
    plt.show()

def write_csv():
    
    '''Write the regression coefficients to a .csv file according with three
       columns: input file name, a, and b corresponding to our solution for 
       that data set.'''
       
    df = pd.DataFrame([(k, k1, v1) for k, v in Coefs.estimated_coefs.items() 
                       for k1, v1 in v.items()], 
                      columns=['File_name', 'Method', 'Coefficients'])
    
    a = [x[0] for x in df['Coefficients'].str[0]]
    a = pd.Series(a)
    df['a'] = a.values
    
    b = [x[1] for x in df['Coefficients'].str[0]]
    b = pd.Series(b)
    df['b'] = b.values
    
    df.drop('Coefficients', axis=1, inplace=True)
    df.to_csv('regression_estimates.csv', sep=',', index=False)  
    
if __name__ == '__main__':

    #------------------------------
    # Load data
    #------------------------------
    data, file_names = load_data.load_data()
    
    #-----------------------------------------------------------------------
    # Aggregate data and make plots for diagnostic tests.
    #------------------------------------------------------------------------
    aggregate_data.residual_plots()
    
    #-------------------------------------------------------------------------
    # Run linear regression with each of the outlined techniques for every 
    # data set and save the resultant coeffients in a nested dictionary.
    #-------------------------------------------------------------------------
    
    run_ols()
    run_irls_huber(tuning_parameter=1.345)
    run_irls_tukey(tuning_parameter=4.685)
    run_wls()
    run_mle_double_exp(num_steps=30000, 
                       alpha=np.array([0., 0.1]),
                       learning_rate=1e-8)
    
    #-------------------------------------------------------------------------
    # Plot data, and the linear fit with each of the outlined techniques.
    #-----------------------------------------------------------------------
    plot()
    
    #-------------------------------------------------------------------------
    # Write the regression coefficients on to a .csv file
    #-----------------------------------------------------------------------
    write_csv()