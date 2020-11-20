#!/usr/bin/env python

##
## Rudimentary analysis (per station) of applying a butterworth low pass filter to remove tides 
## 

import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy import signal
                             
from pandas.tseries.frequencies import to_offset

def makePlot(start, end, station, stationName, df):
    sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird
    # Plot daily, weekly resampled, and 7-day rolling mean time series together
    fig, ax = plt.subplots()
    ax.plot(df.loc[start:end][station],
    marker='.', markersize=1, linewidth=0.1,color='gray',label='OBS-WL')
    ax.plot(df.loc[start:end]['LOWPASS'],
    marker='o', color='green',markersize=1, linestyle='-', linewidth=.5, label='OBS-Pred')
    ax.set_ylabel(r'WL (m) versus MSL')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(station)+'_detided.png')
    plt.close()
    #plt.show()


# Get the basic data 

meta='/projects/sequence_analysis/vol1/prediction_work/ADDA_202010191800_202010231800/obspkl/obs_wl_metadata_202010191800_202010231800.pkl'

fhh='/home/jtilson/ADCIRCSupportTools/get_obs_stations/test/2018/NEWTEST-HH/StationTest/obspkl/obs_wl_detailed_2018-01-01_00:00_2018-12-31_18:00.pkl'

dhh=pd.read_pickle(fhh)
df_meta=pd.read_pickle(meta)
df_meta.set_index('stationid',inplace=True)

stations = df_meta.index.to_list()
#stations=[8534720, 8658163, 8768094]

#############
## Specify the lowpass filter
#############
# This is for hourly data

fs = 1/3600 # Convert samples to Hz: 1pt/6min-bl * 1 6min-bl/360 sec -> 1/10/360 pt/sec (Hz)
nyquist = fs/2 # Standard shift: Hz
#cutoff=0.005 # 10 measurements - seems pretty good
cutoff=0.012 # Keep everything less than a 41.7 hours cutoff
print('cutoff= {} hours'.format(1/cutoff*nyquist*1*3600)) # 41.7 hours
filterOrder=5
b, a = signal.butter(filterOrder, cutoff, btype='lowpass') # Could use fs=None #low pass filter


# Run the plot pipeline

sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird

# Full pipeline testing
#station=8410140
#station=8534720
#station=8658163
#station=8768094
#stations=[8534720]

for station in stations:
    dhh_test = dhh[station].dropna() # Make it as series
    datalow = signal.filtfilt(b, a, dhh_test)    
    dhh_combined = dhh_test.reset_index()
    dhh_combined['LOWPASS']=list(datalow)
    dhh_combined.set_index('date_time',inplace=True)
    start, end = '2018-01', '2018-12'
    stationName = df_meta.loc[int(station)]['stationname']
    makePlot(start,end, station, stationName, dhh_combined) 









