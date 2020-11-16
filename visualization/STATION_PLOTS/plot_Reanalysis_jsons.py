#!/usr/bin/env python

##
## Grab the META data JSON generated from GetObsStations: (meta json)
## Grab the Merged ADC,OBS,ERR time series data JSON computed by compError
##
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas.tseries.frequencies import to_offset
import json

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

def dictToDataFrame(dataDict, src):
    stations = list(dataDict.keys())
    dt = [dataDict[x][src]['TIME'] for x in dataDict.keys()]
    #dx = [dataDict[x][src]['WL'] for x in dataDict.keys()]
    indexes = dt[0]
    df=pd.DataFrame(indexes)
    df.columns=['TIME']
    for station in stations:
        df[station]=dataDict[station][src]['WL']
    df.set_index('TIME',inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

f='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/pipelines/NEWTEST-ALLSTATIONS/adc_obs_error_merged.json'
meta='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/pipelines/NEWTEST-ALLSTATIONS/obs_wl_metadata.json'

# Read jsons, Convert to DataFrames for subsequent plotting

# Metadata
with open(meta, 'r') as fp:
    try:
        metaDict = json.load(fp)
    except OSError:
        utilities.log.error("Could not open/read file {}".format(meta)) 
        sys.exit()

# Timeseries data
with open(f, 'r') as fp1:
    try:
        dataDict = json.load(fp1)
    except OSError:
        utilities.log.error("Could not open/read file {}".format(meta))
        sys.exit()

#
columns = list(dataDict.keys()) # For subsequent naming - are we sure order is maintained?
# Metadata
df_meta=pd.DataFrame(metaDict)
df_meta.set_index('stationid',inplace=True)
# Time series data. This ONLY works on compError generated jsons

dtype=['OBS','ADC','ERR']
stations = list(dataDict.keys())

df_obs_all = dictToDataFrame(dataDict, 'OBS')
df_adc_all = dictToDataFrame(dataDict, 'ADC')
df_err_all = dictToDataFrame(dataDict, 'ERR')

#############
# Run the plot pipeline

sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird

# Full pipeline testing
#station='8410140'
#station='8534720'
#station='8658163'
station='8768094'

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
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
ax.legend(fontsize=10);
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.savefig(station+'.png')
plt.show()








