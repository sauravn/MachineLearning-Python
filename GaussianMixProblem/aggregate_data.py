import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

def load_concatenated_data():
    
    ''' Aggregate data in to a dictionary'''
    
    print "Concatenated loading of data starting ... "
    path = os.path.dirname(os.path.abspath(__file__))
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    print "Finished concatenating data!"
    return concatenated_df

def residual_plots():
    
    ''' Plots the OLS residuals vs predictors. Also plots a QQ plot of 
    residuals'''
    
    plt.style.use('ggplot')
    
    aggregated_data = load_concatenated_data()
    lm_ols = smf.ols(formula='y ~ x', data=aggregated_data).fit() # OLS fit
    
    #---------------------------------------------
    # Scatter plot of OLS residuals vs predictors
    #---------------------------------------------
    #plt.scatter(aggregated_data['x'].values, lm_ols.resid)
    plt.scatter(aggregated_data['x'].values, lm_ols.resid**2)
    plt.xlabel('x')
    plt.ylabel('OLS squared residuals')
    plt.savefig('squared_residuals_scatterplot.pdf')
    plt.show()
    
    #----------------------
    ## QQ plot of residuals
    #----------------------
    fig = sm.qqplot(lm_ols.resid, line='s')
    fig.savefig('residuals_QQplot.pdf')
    plt.show(fig)