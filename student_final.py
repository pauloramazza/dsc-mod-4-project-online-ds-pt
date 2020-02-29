#!/usr/bin/env python
# coding: utf-8

# In[28]:


#Import the necessary libraries for EDA, visualization and time series modelling.
#Data visualization and manipulation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
import warnings
warnings.simplefilter('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Line that is not converging
    likev = mdf.profile_re(0, dist_low=0.1, dist_high=0.1)
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
#Time series analysis tools.
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm


# In[2]:


#Helper Functions
#Obtain dates form the data. Source: Flatiron School
def get_datetimes(df):
    return pd.to_datetime(df.columns.values[1:], format='%Y-%m')

#Convert the data into long format. Source: Flatiron School
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted

def acf_pacf(df,alags=48,plags=48):
    #Create figure
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,8))
    #Make ACF plot
    plot_acf(df,lags=alags, zero=False,ax=ax1)
    #Make PACF plot
    plot_pacf(df,lags=plags, ax=ax2)
    plt.show()

def seasonal_plots(df,N=13,lags=[12,24,36,48,60,72]):
    #Differencing the rolling mean to find seasonality in the resulting acf plot.
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,8))
    rolling = TS_70808 - TS_70808.rolling(N).mean()
    plot_acf(rolling.dropna(),lags=lags,ax=ax1)
    plot_pacf(rolling.dropna(),lags=lags,ax=ax2)
    plt.show();
    
def train_test(df):
    #Set trainning dsata before 2016
    train = df[:'2015-04']
    #Set test data starting 2016
    test = df['2015-05':]
    return train, test

def model_fit(df,pdq=(1,0,1),pdqs=(0,0,0,1)):
    train, test = train_test(df)
    model = SARIMAX(train,order=pdq,seasonal_order=pdqs)
    results = model.fit()
    results.summary
    residuals = results.resid
    print(results.summary())
    results.plot_diagnostics(figsize=(11,8))
    plt.show();
    return train, test, results

def test_RMSE(df,pdq=(1,0,1),pdqs=(0,0,0,1), display=True):
    X = df.values
    train, test = X[:-36],X[-36:]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = SARIMAX(history, order=pdq,seasonal_order=pdqs)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test[t])
    rmse = sqrt(MSE(test, predictions))
    print('SARIMA model RMSE on test data: %.5f' % rmse)
    if display:
        plt.figure(figsize=(13,6))
        plt.title('Actual Test Data vs. Predictions')
        plt.plot(history[-36:],label='Actual', color='b')
        plt.plot(predictions,label='Predictions',color='r')
        plt.legend(loc='best')
        plt.show()

def train_RMSE(train, results, display = True):
    train_pred = results.predict(-36)
    rmse = sqrt(MSE(train[-36:],train_pred))
    print(f'SARIMA model RMSE on train data: %.5f' % rmse)
    if display:
        plt.figure(figsize=(13,6))
        train[-60:].plot(label='Actual',color='b')
        train_pred.plot(label='Predicted',color='r')
        plt.legend(loc='best')
        plt.title('Actual Train Data vs. Predicted Returns')
        plt.show()

def forecast_model(df,pdq=(1,0,1),pdqs=(0,0,0,1), display=True,zc='input zipcode'):
    model = SARIMAX(df, order=pdq,seasonal_order=pdqs)
    model_fit = model.fit()
    output = model_fit.get_prediction(start='2018-04',end='2028-04', dynamic=True)
    forecast_ci = output.conf_int()
    if display:
        fig, ax = plt.subplots(figsize=(13,6))
        output.predicted_mean.plot(label='Forecast')
        ax.fill_between(forecast_ci.index,forecast_ci.iloc[:, 0],forecast_ci.iloc[:, 1],
                        color='k', alpha=.25,label='Conf Interval')
        plt.title('Forecast of Monthly Returns')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()
    year_1= (1+output.predicted_mean[:12]).prod()-1
    year_3=(1+output.predicted_mean[:36]).prod()-1
    year_5= (1+output.predicted_mean[:60]).prod()-1
    year_10=(1+output.predicted_mean).prod()-1
    print(f'Total expected return in 1 year: {round(year_1*100,2)}%')
    print(f'Total expected return in 3 years: {round(year_3*100,2)}%')
    print(f'Total expected return in 5 year: {round(year_5*100,2)}%')
    print(f'Total expected return in 10 years: {round(year_10*100,2)}%')
    tot_ret = [zc,year_1,year_3,year_5,year_10]
    return tot_ret


# In[7]:


df_zillow = pd.read_csv('zillow_data.csv')
print(df_zillow.info(),'\n')
print(f'Unique zip codes: {df_zillow.RegionName.nunique()}')
df_zillow.head()


# In[4]:


cd


# In[6]:


cd dsc-mod-4-project-online-ds-sp-000


# In[8]:


#Get zipcodes with a size rank in the top 20% (highly urbanized zipcodes).
print(df_zillow.SizeRank.describe(),'\n')
#Calculate the 20% cutoff value.
sr_20 = df_zillow.SizeRank.quantile(q=0.20)
print(f'Size Rank 20% cutoff value: {sr_20}')
#Get data frame with selected zipcodes. Keep values and zipcodes only.
zc_top20= df_zillow[df_zillow['SizeRank']<sr_20].drop(['RegionID','City','State','Metro','CountyName','SizeRank'],axis=1)
print(f'Amount of zipcodes: {len(zc_top20)}')


# In[9]:


zc_top20['yr_avg']=zc_top20.iloc[:,-12:].mean(skipna=True, axis=1)
#Get zipcodes with an average value 1 decile above the median and 1.5 deciles below.
print(zc_top20['yr_avg'].describe(),'\n')
#Calculate the 60% cutoff value (1 decile above).
q_60 = zc_top20['yr_avg'].quantile(q=0.60)
print(f'Average Value 60% cutoff value: {round(q_60,2)}')
#Calculate the 35% cutoff value (1.5 deciles below).
q_35 = zc_top20['yr_avg'].quantile(q=0.35)
print(f'Average Value 35% cutoff value: {round(q_35,2)}')
#Get data frame with selected zipcodes.
zc_pref= zc_top20[(zc_top20['yr_avg']<q_60) & (zc_top20['yr_avg']>q_35)]
print(f'Amount of zipcodes: {len(zc_pref)}')


# In[10]:


#Calculate historical return on investment
zc_pref['ROI']= (zc_pref['2018-04']/zc_pref['1996-04'])-1
#Calculate standard deviation of monthly values
zc_pref['std']=zc_pref.loc[:,'1996-04':'2018-04'].std(skipna=True, axis=1)
#Calculate historical mean value
zc_pref['mean']=zc_pref.loc[:,'1996-04':'2018-04'].mean(skipna=True, axis=1)
#Calculate coefficient of variance
zc_pref['CV']=zc_pref['std']/zc_pref['mean']
#Show calculated values
zc_pref[['RegionName','std','mean','ROI','CV']].head()


# In[11]:


#Descriptive statistics of coefficients of variance.
print(zc_pref.CV.describe())
#Define upper limit of CV according to risk profile.
upper_cv = zc_pref.CV.quantile(.6)
print(f'\nCV upper limit: {upper_cv}')
#Get the 5 zipcodes with highest ROIs within the firms risk profile.
zc_best5 = zc_pref[zc_pref['CV']<upper_cv].sort_values('ROI',axis=0,ascending=False)[:5]
print('\n Best 5 Zipcodes:')
zc_best5[['RegionName','ROI','CV']]


# In[12]:


#Get Location Names
best5_zipcodes = list(zc_best5.RegionName.values)
for i in best5_zipcodes:
    city = df_zillow[df_zillow['RegionName']==i].City.values[0]
    state = df_zillow[df_zillow['RegionName']==i].State.values[0]
    print(f'Zipcode : {i} \nLocation: {city}, {state}\n')


# In[13]:


TS_zc5 = zc_best5.drop(['yr_avg','std','mean','ROI','CV'],axis=1)
TS_zc5 = melt_data(TS_zc5).set_index('time')
print('Time series data for the 5 zipcodes:\n',TS_zc5.head())
#Create individualized time series for each zipcode.
#List containing the 5 different time series.
dfs_ts = []
for zc in TS_zc5.RegionName.unique():
    #Create separate dataframes for each zipcode with a monthly frequency.
    df = TS_zc5[TS_zc5['RegionName']==zc].asfreq('MS')
    dfs_ts.append(df)
print('\nZipcode 70808 time series:')
dfs_ts[0].head()


# In[14]:


for i in range(len(dfs_ts)):
    print(f'Value descriptive statistics for zipcode {dfs_ts[i].RegionName[0]}:')
    print(f'{dfs_ts[i].value.describe()}\n')


# In[15]:


for i in range(5):
    dfs_ts[i].value.plot(label=dfs_ts[i].RegionName[0],figsize=(15,6))
    plt.legend()


# In[16]:


#Calculate monthly returns in new column 'ret' for each zipcode.
for zc in range(len(dfs_ts)):
    dfs_ts[zc]['ret']=np.nan*len(dfs_ts[zc])
    for i in range(len(dfs_ts[zc])-1):
        dfs_ts[zc]['ret'][i+1]= (dfs_ts[zc].value.iloc[i+1] / dfs_ts[zc].value.iloc[i]) - 1


# In[17]:


#Plot the monthly returns of each zipcode
for i in range(len(dfs_ts)):
    dfs_ts[i].ret.plot(figsize=(11,5), color = 'b')
    plt.title(f'Zipcode: {dfs_ts[i].RegionName[0]}')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend(loc='best')
    plt.show()


# In[18]:



#Plot each of the zipcodes' returns with their respective rolling mean and rolling standard deviation.
#Vizually test for stationarity.
for i in range(len(dfs_ts)):
    rolmean = dfs_ts[i].ret.rolling(window = 12, center = False).mean()
    rolstd = dfs_ts[i].ret.rolling(window = 12, center = False).std()
    fig = plt.figure(figsize=(11,5))
    orig = plt.plot(dfs_ts[i].ret, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(f'Rolling Mean & Standard Deviation for Zipcode: {dfs_ts[i].RegionName[0]}')
    plt.show()


# In[21]:


for i in range(5):
    results = adfuller(dfs_ts[i].ret.dropna())
    print(f'ADFuller test p-value for zipcode: {dfs_ts[i].RegionName[0]}')
    print('p-value:',results[1])
    if results[1]>0.05:
        print('Fail to reject the null hypothesis. Data is not stationary.\n')
    else:
        print('Reject the null hypothesis. Data is stationary.\n')


# In[ ]:


#Time series analysis tools.
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import Pmdarima as pm


# In[38]:


for i in [1,2,4]:
    #Perform adfuller test and drop NaN values created when calculating monthly returns.
    results = adfuller(dfs_ts[i].ret.diff().dropna())
    print(f'ADFuller test p-value for zipcode: {dfs_ts[i].RegionName[0]}')
    print('p-value:',results[1])
    if results[1]>0.05:
        print('Fail to reject the null hypothesis. Data is not stationary.\n')
    else:
        print('Reject the null hypothesis. Data is stationary.\n')


# In[39]:


#Instantiate individual time series for each of the zipcodes.
TS_70808 = dfs_ts[0].ret.dropna()#Zipcode 70808 monthly returns time series

TS_29461 = dfs_ts[1].ret.dropna()#Zipcode 29461 monthly returns time series
TS_29461d = dfs_ts[1].ret.diff().dropna()#Zipcode 29461 monthly returns differenced time series

TS_3820 = dfs_ts[2].ret.dropna()#Zipcode 3820 monthly returns time series
TS_3820d = dfs_ts[2].ret.diff().dropna()#Zipcode 3820 monthly returns differenced time series

TS_52722 = dfs_ts[3].ret.dropna()#Zipcode 52722 monthly returns time series

TS_70809 = dfs_ts[4].ret.dropna()#Zipcode 70809 monthly returns time series
TS_70809d = dfs_ts[4].ret.diff().dropna()#Zipcode 70809 monthly returns differenced time series


# In[32]:


acf_pacf(TS_29461d)


# In[40]:


seasonal_plots(TS_29461d, N=13)


# In[50]:


import pmdarima as pm


# In[36]:


get_ipython().system('pip install pmdarima ')


# In[ ]:


#Use auto arima function to find the best non-seasonal and seasonal parameters to fit the model.
results = pm.auto_arima(TS_29461,information_criterion='aic',d=1,
                        stepwise=True,trace=True,error_action='ignore',suppress_warnings=True)
results


# In[45]:


pip freeze


# In[47]:


import sys
print(sorted(sys.path))


# In[49]:


get_ipython().system('pip3 install pmdarima')


# In[51]:


#Fit the SARIMA model and get results.
pdq = (2,1,2)
pdqs = (0,0,0,1)
train, test, results = model_fit(TS_29461,pdq=pdq,pdqs=pdqs)


# In[52]:


train_RMSE(train, results)


# In[53]:


test_RMSE(TS_29461,pdq=pdq,pdqs=pdqs, display=True)


# In[54]:


ret_29461=forecast_model(TS_29461,pdq=pdq,pdqs=pdqs,zc=29461)


# In[ ]:




