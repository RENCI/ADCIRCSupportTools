#!/usr/bin/env python

##
## Grab the META data PKL generated from GetObsStations: (metapkl)
## Grab the Merged ADC,OBS,ERR time series data CSV computed by (mergedf) 
## Specify list up to four stations for comparison
##
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.frequencies import to_offset

# The means are proper means for the month./week. The loffsey simply changes the index underwhich it will be stored
# Moved away from deprecated codes
# THe weekly means index at starting week +3 days.
def station_level_means(df_obs, df_adc, df_err, station):
    dfs = pd.DataFrame()
    dfs['OBS']=df_obs[station]
    dfs['ADC']=df_adc[station]
    dfs['ERR']=df_err[station]
    #
    dfs['Year'] = dfs.index.year
    dfs['Month'] = dfs.index.month
    dfs['Hour'] = dfs.index.hour
    #
    data_columns = ['OBS','ADC','ERR']
    # Resample to monthly frequency, aggregating with mean
    # dfs_monthly_mean = dfs[data_columns].resample('MS',loffset=pd.Timedelta(15,'d')).mean() #MS restarts meaning from the start not the end
    # dfs_weekly_mean = dfs[data_columns].resample('W',loffset=pd.Timedelta(84,'h')).mean()
    dfs_monthly_mean = dfs[data_columns].resample('MS').mean() #MS restarts meaning from the start not the end
    dfs_monthly_mean.index = dfs_monthly_mean.index - to_offset("15d")
    dfs_weekly_mean = dfs[data_columns].resample('W').mean()
    # No need for this...dfs_weekly_mean.index = dfs_weekly_mean.index+to_offset("84h")
    # Get rolling
    dfs_7d = dfs[data_columns].rolling(7, center=True).mean()
    return dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d

f='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/pipelines/NEWTEST-ALLSTATIONS/adc_obs_error_merged.csv'
meta='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/pipelines/NEWTEST-ALLSTATIONS/obs_wl_metadata.pkl'

## Start the work

df = pd.read_csv(f,header=0, index_col=0)
df.index = pd.to_datetime(df.index)
df_meta=pd.read_pickle(meta)
df_meta.set_index('stationid',inplace=True)
df_adc_all = df[df['SRC']=='ADC'].copy()
df_obs_all = df[df['SRC']=='OBS'].copy()
df_err_all = df[df['SRC']=='ERR'].copy()
df_adc_all.drop('SRC',inplace=True,axis=1)
df_obs_all.drop('SRC',inplace=True,axis=1)
df_err_all.drop('SRC',inplace=True,axis=1)

# Now choose a single station and merge OBS,ADC,ERR together
station='8410140'
#dfs = pd.DataFrame()
#dfs['OBS']=df_obs_all[station]
#dfs['ADC']=df_adc_all[station]
#dfs['ERR']=df_err_all[station]
#dfs['Year'] = dfs.index.year
#dfs['Month'] = dfs.index.month
#dfs['Hour'] = dfs.index.hour

dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d = station_level_means(df_obs_all, df_adc_all, df_err_all, station)

# https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
# 1
sns.set(rc={'figure.figsize':(11, 4)})
dfs['ADC'].plot(linewidth=0.5)

# 2
cols_plot=['OBS','ADC','ERR']
axes = dfs[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('WL (m)')

# 3
import matplotlib.dates as mdates
fig, ax = plt.subplots()
ax.plot(dfs.loc['2018-01-01':'2018-03-01', 'ERR'], marker='o', linestyle='-')
ax.set_ylabel('Error (m)')
ax.set_title('Jan-Mar 2018 Water level diffs (m)')
# Set x-axis major ticks to weekly interval, on Mondays
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
# Format x-tick labels as 3-letter month name and day number
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));

#4 
# seasonality NEED to accopunt for the monthly flanking
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['OBS', 'ADC', 'ERR'], axes):
    sns.boxplot(data=dfs, x='Month', y=name, ax=ax)
    ax.set_ylabel('m')
    ax.set_title(name)
    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')

#5
# Subsampling and means
#data_columns = ['OBS','ADC','ERR']

# Resample to monthly frequency, aggregating with mean
#dfs_monthly_mean = dfs[data_columns].resample('M').mean()
#dfs_weekly_mean = dfs[data_columns].resample('W').mean()
#dfs_monthly_mean.head(3)

# Get rolling

#dfs_7d = dfs[data_columns].rolling(7, center=True).mean()
#dfs_7d.head(10)

#6 Obs data with monthly means
start, end = '2018-01', '2018-06'
# Plot daily and monthly resampled time series together
fig, ax = plt.subplots()
ax.plot(dfs.loc[start:end, 'ADC'],
marker='.', color='gray', markersize=1, linewidth=0.1, label='Daily')
ax.plot(dfs_monthly_mean.loc[start:end, 'ADC'],
marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('water level (m)')
ax.legend();

#6b try weekly
start, end = '2018-01', '2018-06'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(dfs.loc[start:end, 'ADC'],
marker='.', color='gray', markersize=1, linewidth=0.1, label='Hourly')
ax.plot(dfs_weekly_mean.loc[start:end, 'ADC'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('water level (m)')
ax.legend();


#7 rolling means with extra meta data`
station='8658163'
stationName = df_meta.loc[int(station)]['stationname']
start, end = '2018-01', '2018-12'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(dfs.loc[start:end, 'ERR'],
marker='.', markersize=1, linewidth=0.1,color='gray',label='Hourly')
ax.plot(dfs_7d.loc[start:end, 'ERR'],
color='red', alpha=0.3, linewidth=.5, linestyle='-', label='7-d Rolling Mean')
ax.plot(dfs_weekly_mean.loc[start:end, 'ERR'],
marker='o', color='green',markersize=6, linestyle='-', label='Weekly Mean')
ax.plot(dfs_monthly_mean.loc[start:end, 'ERR'],
color='black',linewidth=0.5, linestyle='-', label='Monthly Mean')
ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})

ax.legend();
#plt.savefig(station+'.png')
plt.show()


# Full pipeline testing
#station='8410140'
#station='8534720'
#station='8658163'
station='8768094'

df = pd.read_csv(f,header=0, index_col=0)
df.index = pd.to_datetime(df.index)
df_meta=pd.read_pickle(meta)
df_meta.set_index('stationid',inplace=True)
df_adc_all = df[df['SRC']=='ADC']
df_obs_all = df[df['SRC']=='OBS']
df_err_all = df[df['SRC']=='ERR']
df_adc_all.drop('SRC',inplace=True,axis=1)
df_obs_all.drop('SRC',inplace=True,axis=1)
df_err_all.drop('SRC',inplace=True,axis=1)

dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d = station_level_means(df_obs_all, df_adc_all, df_err_all, station)

stationName = df_meta.loc[int(station)]['stationname']
start, end = '2018-01', '2018-12'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(dfs.loc[start:end, 'ERR'],
marker='.', markersize=1, linewidth=0.1,color='gray',label='Hourly')
ax.plot(dfs_7d.loc[start:end, 'ERR'],
color='red', alpha=0.3, linewidth=.5, linestyle='-', label='7-d Rolling Mean')
ax.plot(dfs_weekly_mean.loc[start:end, 'ERR'],
marker='o', color='green',markersize=6, linestyle='-', label='Weekly Mean')
ax.plot(dfs_monthly_mean.loc[start:end, 'ERR'],
color='black',linewidth=0.5, linestyle='-', label='Monthly Mean')
ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
ax.legend();
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(station+'.png')
#plt.show()








