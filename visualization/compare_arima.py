#!/usr/bin/env python
##
## Compare err timeseries, to the ARIMA fits and add horizontal lines to 
## reflect ADDA average vs arima predicted value for use by kriging
##
##
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plotEstimates(station, stationName, df_time, df_predict, aveValue, prdValue):
    maxValue = df_time.max()
    minValue = df_time.min()
    sns.set(rc={'figure.figsize':(11, 4)})
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_time.loc[start:end],
    marker='.', markersize=5, linewidth=0.8,color='gray',label='ADDA error')
    ax.plot(df_predict.loc[start:end],
    color='blue', marker='.',alpha=0.8, linewidth=.8, linestyle='-', label='ARIMA prediction')
    plt.axhline(y=aveValue, linewidth=.8, color='r', linestyle='--', label='ADDA Ave estimate')
    plt.axhline(y=prdValue, linewidth=.8, color='black', linestyle='--', label='ARIMA predicted estimate')
    ax.legend(loc ='lower left',ncol=3, fontsize = 6)
    ax.set_ylabel(r'$\Delta$ WL (m) versus MSL, ADC-OBS.')

    ax.set_ylim([minValue, maxValue]) #-0.4, 0.1])

    ax.set_xlabel('')
    ax.get_xaxis().set_visible(True)
    ax.set_title(stationName+' Error profile', fontdict={'fontsize': 10, 'fontweight': 'medium'})
    hours = mdates.HourLocator(interval = 6)
    h_fmt = mdates.DateFormatter('%m-%d-%H')
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(station)+'-ARIMA.png')
    plt.close()

maindir='/home/jtilson/ADCIRCSupportTools/ADDA/ARIMA-TESTING'
metadata='202102151200_202102191200'

# Basic station metadata:For plot annotation
fmeta=maindir+'/ADDA/ADDA_'+metadata+'/obspkl/obs_water_level_metadata_'+metadata+'.pkl'


# ADDA data
ftimeADDA=maindir+'/ADDA/ADDA_'+metadata+'/errorfield/tideTimeErrors_'+metadata+'.pkl'
fmergedADDA=maindir+'/ADDA/ADDA_'+metadata+'/errorfield/adc_obs_error_merged_'+metadata+'.csv'
fsummaryADDA=maindir+'/ADDA/ADDA_'+metadata+'/errorfield/stationSummaryAves_'+metadata+'.csv'

# ARIMA data
ftimeARIMA=maindir+'/ARIMA/ADDA_'+metadata+'/errorfield-ARIMA/tideTimeErrors_'+metadata+'.pkl'
fmergedARIMA=maindir+'/ARIMA/ADDA_'+metadata+'/errorfield-ARIMA/adc_obs_error_merged_'+metadata+'.csv'
fsummaryARIMA=maindir+'/ARIMA/ADDA_'+metadata+'/errorfield-ARIMA/stationSummaryAves_'+metadata+'.csv'
fpredictedARIMA=maindir+'/ARIMA/ADDA_'+metadata+'/errorfield-ARIMA/ARIMApredictedErrorsARIMA.pkl'

# Get metadata 
df_meta=pd.read_pickle(fmeta)
df_meta.set_index('stationid',inplace=True)

# Get ADDA data
df_timeADDA=pd.read_pickle(ftimeADDA)
df_mergedADDA=pd.read_csv(fmergedADDA,index_col=0, header=0)
df_summaryADDA=pd.read_csv(fsummaryADDA,index_col=0, header=0)

# Get ARIMA data
df_timeARIMA=pd.read_pickle(ftimeARIMA)
df_mergedARIMA=pd.read_csv(fmergedARIMA,index_col=0, header=0)
df_summaryARIMA=pd.read_csv(fsummaryARIMA,index_col=0, header=0)
df_predictedARIMA=pd.read_pickle(fpredictedARIMA)

# Get time ranges for plotting - for now just get them all 
start=df_timeADDA.index.min()
end=df_timeADDA.index.max()

# Get list of stations. ARIMA most likely dropped a computing summaries
stations = df_summaryARIMA.dropna().index.to_list()

##
## Test a single plot
##

for station in stations:
    #station=8658163
    print('Plot station {}'.format(station))
    stationName=df_meta.loc[station,'stationname']
    df_time=df_timeADDA[station]
    df_predict=df_predictedARIMA[station]
    aveValue=df_summaryADDA.loc[station,'mean']
    prdValue=df_summaryARIMA.loc[station,'mean']
    plotEstimates(station, stationName, df_time, df_predict, aveValue, prdValue)


##
## Do a plot for each station
##
