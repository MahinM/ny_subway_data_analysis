import scipy.stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ggplot import * 

#Read in data
TW = pd.read_csv('improved_TW.csv')

TW.shape
#42,549 rows, 27 columns

TW_rain = TW[TW['rain'] == 1]
#9585 rows
TW_no_rain = TW[TW['rain'] == 0]
#33064 rows
### 77.7% of days are not rainy

#visualize the entries on rainy and non-rainy days
plt.figure()
TW_no_rain['ENTRIESn_hourly'].hist(histtype='bar', bins=20, color = 'blue', label='Rain', range=(0,20000))
plt.show()

plt.figure()
TW_rain['ENTRIESn_hourly'].hist(histtype='bar',bins=20, color='green', label='No Rain', range=(0,20000))
plt.show()

#formal test for normality
#Anderson-Darling
A, critical, sig = scipy.stats.anderson(TW['ENTRIESn_hourly'], dist='norm')

#Tests null hypothesis that the two populations are the same.
U,p = scipy.stats.mannwhitneyu(TW_no_rain['ENTRIESn_hourly'],TW_rain['ENTRIESn_hourly'])
########## Problem: p comes back as nan
####Used reference to compute p-value: http://stats.stackexchange.com/questions/116315/problem-with-mann-whitney-u-test-in-scipy
#Double p for 2-tailed test
p_two_tailed = p * 2

#Check means
np.mean(TW['ENTRIESn_hourly'])
#1886.59

np.mean(TW_no_rain['ENTRIESn_hourly'])
#1845.54

np.mean(TW_rain['ENTRIESn_hourly'])
#2028.20


pandas.options.mode.chained_assignment = None

TW['day_of_week'] = [datetime.datetime.strftime(x,'%A') 
						for x in pd.to_datetime(TW['datetime'])]
    
tw_day_of_week = TW[['day_of_week','ENTRIESn_hourly']].groupby('day_of_week', as_index=False).sum()
        
    
plot = ggplot(tw_day_of_week, aes('day_of_week'))  \
        + geom_bar(aes(weight='ENTRIESn_hourly'), fill='blue') \
        + ggtitle('NYC Subway Ridership by Day of Week') + xlab('Day') + ylab('Riders')

###############################################################################

###Linear Regression


def linear_regression(features, values):
	features = sm.add_constant(features)
	model = sm.OLS(values, features)
	results = model.fit()
	return results

'''
[u'UNIT', u'DATEn', u'TIMEn', u'ENTRIESn', u'EXITSn', u'ENTRIESn_hourly',
       u'EXITSn_hourly', u'datetime', u'hour', u'day_week', u'weekday',
       u'station', u'latitude', u'longitude', u'conds', u'fog', u'precipi',
       u'pressurei', u'rain', u'tempi', u'wspdi', u'meanprecipi',
       u'meanpressurei', u'meantempi', u'meanwspdi', u'weather_lat',
       u'weather_lon', u'day_of_week']
'''
TW['day_of_week_num'] = [datetime.datetime.strftime(x,'%w') 
						for x in pd.to_datetime(TW['datetime'])]
    


features = TW[['hour', 'rain']]
values = TW['ENTRIESn_hourly']

results = linear_regression(features,values)

features = TW[['hour', 'rain', 'day_week']]
dummy_station = pd.get_dummies(TW['station'], prefix = 'station')
dummy_station = dummy_station.drop('station_1 AVE', 1)
features = features.join(dummy_station)

results = linear_regression(features,values)

features = TW[['hour', 'rain', 'day_week', 'latitude', 'longitude', 'fog', 'precipi', 'pressurei', 'tempi', 'wspdi']]
results = linear_regression(features,values)


features = TW[['hour', 'rain', 'day_week', 'latitude', 'longitude', 'fog', 'precipi', 'pressurei', 'tempi', 'wspdi']]
dummy_units = pd.get_dummies(TW['UNIT'], prefix = 'UNIT')
features = features.join(dummy_units)

results = linear_regression(features,values)


features = TW[['hour', 'rain', 'day_week']]
dummy_units = pd.get_dummies(TW['UNIT'], prefix = 'UNIT')
features = features.join(dummy_units)

results = linear_regression(features,values)

features = TW[['hour','rain', 'day_week','pressurei', 'tempi']]
results = linear_regression(features,values)
dummy_conds = pd.get_dummies(TW['conds'], prefix = 'conds')
features = features.join(dummy_conds)

dummy_station = pd.get_dummies(TW['station'], prefix = 'station')
dummy_station = dummy_station.drop('station_1 AVE', 1)
features = features.join(dummy_station)
results = linear_regression(features,values)


features = TW[['hour','rain', 'day_week','pressurei']]

dummy_station = pd.get_dummies(TW['station'], prefix = 'station')
dummy_station = dummy_station.drop('station_1 AVE', 1)
features = features.join(dummy_station)

results = linear_regression(features,values)


#####Probability plot
plt.figure()
scipy.stats.probplot(TW['ENTRIESn_hourly'] - results.predict(), plot=plt)
plt.show()

#####Residual plot
plt.figure()
plt.plot(TW['ENTRIESn_hourly'] - results.predict())
plt.show()

