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
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter
import datetime as dt
from utilities.utilities import utilities

def butter_lowpass_filter(df_data,filterOrder=10, numHours=200):
    """
    Note. Need to drop nans from the data set
    """
    numHoursCutoff=numHours
    cutoff = 2/numHoursCutoff #Includes nyquist adjustment
    #ddata=df[station]
    ddata = df_data
    print('cutoff={} Equiv number of hours is {}'.format(cutoff, 2/cutoff))
    print('filterOrder is {}'.format(filterOrder))
    sos = signal.butter(filterOrder, cutoff, 'lp', output='sos',analog=False)
    datalow = signal.sosfiltfilt(sos, ddata)
    #print('ddata {}'.format(ddata))
    #print('LP data {}'.format(datalow))
    return datalow

def fft_lowpass(signal, lowhrs):
    """
    Performs a low pass filer on the series.
    low and high specifies the boundary of the filter.
    >>> from oceans.filters import fft_lowpass
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(500)  # Time in hours.
    >>> x = 2.5 * np.sin(2 * np.pi * t / 12.42)
    >>> x += 1.5 * np.sin(2 * np.pi * t / 12.0)
    >>> x += 0.3 * np.random.randn(len(t))
    >>> filtered = fft_lowpass(x, low=1/30, high=1/40)
    >>> fig, ax = plt.subplots()
    >>> l1, = ax.plot(t, x, label='original')
    >>> l2, = ax.plot(t, filtered, label='filtered')
    >>> legend = ax.legend()
    NOTE for high freq we always take the highesyt possible
    which is just the sampling freq
    """
    low = 1/lowhrs
    high = 1/signal.shape[0]
    print('FFT: low {}, high {}'.format(low,high))
    if len(signal) % 2:
        result = np.fft.rfft(signal, len(signal))
    else:
        result = np.fft.rfft(signal)
    freq = np.fft.fftfreq(len(signal))[:len(signal) // 2 + 1]
    factor = np.ones_like(freq)
    factor[freq > low] = 0.0
    sl = np.logical_and(high < freq, freq < low)
    a = factor[sl]
    # Create float array of required length and reverse.
    a = np.arange(len(a) + 2).astype(float)[::-1]
    # Ramp from 1 to 0 exclusive.
    a = (a / a[0])[1:-1]
    # Insert ramp into factor.
    factor[sl] = a
    result = result * factor
    return np.fft.irfft(result, len(signal))

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
    dfs_monthly_mean.index = dfs_monthly_mean.index + to_offset("15d")
    dfs_weekly_mean = dfs[data_columns].resample('W').mean()
    # No need for this...dfs_weekly_mean.index = dfs_weekly_mean.index+to_offset("84h")
    # Get rolling
    dfs_7d = dfs[data_columns].rolling(7, center=True).mean()
    return dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d

# Brian prefers to use a lowpass filter. than the means


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

def makePlot(start, end, station, src, stationName, dfs, dfs_7d, dfs_weekly_mean, dfs_monthly_mean): 
    sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird
    # Plot daily, weekly resampled, and 7-day rolling mean time series together
    fig, ax = plt.subplots()
    ax.plot(dfs.loc[start:end, src],
    marker='.', markersize=1, linewidth=0.1,color='gray',label='Hourly')
    ax.plot(dfs_7d.loc[start:end, src],
    color='red', alpha=0.3, linewidth=.5, linestyle='-', label='7-d Rolling Mean')
    ax.plot(dfs_weekly_mean.loc[start:end, src],
    marker='o', color='green',markersize=6, linestyle='-', label='Weekly Mean')
    ax.plot(dfs_monthly_mean.loc[start:end, src],
    color='black',linewidth=0.5, linestyle='-', label='Monthly Mean')
    ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(station+'.png')
    plt.close()
    #plt.show()

def makeLowpassPlot(start, end, lowpassAllstations, filterOrder='', metadata=['lowpass','LP']):
    """
    An entry the dict (lowpassAllStations) carries all ther OBS,DIFF,LP data sets
    Plot the OBS and Diff data in a prominant way. Then layer on the cutoffs
    """
    colMetadata = metadata[1]
    nameMetadata = metadata[0]
    plotterDownmove=0.6
    station=lowpassAllstations['station']
    stationName=lowpassAllstations['stationName']
    print('station {} Name {}'.format(station, stationName))
    #
    plt.close()
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
    fig, ax = plt.subplots()
    # OBS and DIFF
    ax.plot(lowpassAllstations[station]['OBS'][start:end],
    marker='.', markersize=1, linewidth=0.1,color='gray',label='Obs Hourly')
    ax.plot(lowpassAllstations[station]['ERR'][start:end],
    color='black', marker='o',markersize=2, linewidth=.5, linestyle='-', label='ADCIRC-OBS')
    # Now start with the lowpass filters
    df_lp = lowpassAllstations[station][colMetadata]
    cutoffs=df_lp.columns.to_list()
    shiftDown=0.0
    for cutoff in cutoffs:
        shiftDown+=plotterDownmove
        ax.plot(df_lp[cutoff][start:end]-shiftDown,
        marker='x', markersize=2, linewidth=0.1,label='_'.join([nameMetadata,cutoff]))
    #
    ax.set_ylabel('WL (m) versus MSL')
    if filterOrder=='':
        ax.set_title(stationName+'. FFT', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    else:
        ax.set_title(stationName+'. Polynomial Order='+str(filterOrder), fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fileMetaname = '_'.join([str(station),nameMetadata])
    fileName = fileMetaname+'.png' if filterOrder=='' else fileMetaname+'_'+str(filterOrder)+'.png'
    plt.savefig(fileName)
    #plt.show()

f='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/adc_obs_error_merged.json'
meta='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/obs_water_level_metadata.json'

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

stations = list(dataDict.keys()) # For subsequent naming - are we sure order is maintained?

# Metadata
df_meta=pd.DataFrame(metaDict)
df_meta.set_index('stationid',inplace=True)

# Time series data. This ONLY works on compError generated jsons
df_obs_all = dictToDataFrame(dataDict, 'OBS').loc['2018-01-01 01:00:00':'2018-12'] # Get whole year inclusive
df_adc_all = dictToDataFrame(dataDict, 'ADC').loc['2018-01-01 01:00:00':'2018-12']
df_err_all = dictToDataFrame(dataDict, 'ERR').loc['2018-01-01 01:00:00':'2018-12']

#############
# Run the plot pipeline

sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird

for station in stations:
    dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d = station_level_means(df_obs_all, df_adc_all, df_err_all, station)
    #start=dfs.index.min().strftime('%Y-%m')
    #end=dfs.index.max().strftime('%Y-%m')
    start, end = '2018-01', '2019-01'
    stationName = df_meta.loc[int(station)]['stationname']
    makePlot(start, end, station, 'ERR', stationName, dfs, dfs_7d, dfs_weekly_mean, dfs_monthly_mean) 

# Additional plots over several lowpass cutoffs

plot_timein=start
plot_timeout=end

# FFT Lowpass 
upshift=4
#hourly_cutoffs=[12,24,48,168,720]
hourly_cutoffs=[12,24,48,168]
cutoffs = [x + upshift for x in hourly_cutoffs]
intersectedStations=stations
fftAllstations=dict()

# FFT Lowpass plots ar several cutoffs

for station in intersectedStations:
    print('Process station {}'.format(station))
    stationName = df_meta.loc[int(station)]['stationname']
    fftdata = dict() # Carry all stations in the order processed buty first add the OBS and detided
    fftdata['OBS']=df_obs_all[station] # Data to interpret
    fftdata['ERR']=df_err_all[station] # Actual detided data set
    df_fft=pd.DataFrame()
    for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
        print('Process cutoff {} for station {}'.format(cutoff,station))
        df_temp = df_err_all[station].dropna()
        df_fft[str(cutoff)]=fft_lowpass(df_temp,lowhrs=cutoffflank)
    df_fft.index = df_temp.index
    fftdata['FFT']=df_fft
    fftAllstations[station]=fftdata
    fftAllstations['station']=station
    fftAllstations['stationName']=stationName
    # For each station plot. OBS,explicit detided, cutoffs
    makeLowpassPlot(plot_timein, plot_timeout, fftAllstations, filterOrder='', metadata=['lowpass_fft','FFT'])

# COnstruct new .csv files at a single cutoff
# FFT Lowpass each station for all time. Then, extract values for all stations every mid week.

upshift=4
hourly_cutoffs=[168]
cutoffs = [x + upshift for x in hourly_cutoffs]
intersectedStations=stations
fftAllstations=dict()

df_err_all_lowpass=pd.DataFrame()
for station in intersectedStations:
    print('Process station {}'.format(station))
    stationName = df_meta.loc[int(station)]['stationname']
    df_fft=pd.DataFrame()
    for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
        print('Process cutoff {} for station {}'.format(cutoff,station))
        df_temp = df_err_all[station].dropna()
        df_fft[str(cutoff)]=fft_lowpass(df_temp,lowhrs=cutoffflank)
    df_fft.index = df_temp.index
    df_err_all_lowpass[station]=df_fft[str(cutoff)]

# Now pull out weekly data starting at the middle of the first week
# Build a list of indexes from which to extract data. Start at the first midweek then increment every 164 hours

starttime='2018-01-03 12:00:00'

listdata = pd.date_range(starttime, periods=52, freq=pd.offsets.Hour(n=168)).values
df_err_all_lowpass_subselect=df_err_all_lowpass.loc[listdata]

# Now process the Rows and build a new datafile for each
# Leverage the df_meta object to create the final datasets
# df_meta and df report stationids as diff types. Yuk.
# Store the list iof filenames into a dict for krig processing

rootdir='.'
subdir='LOWPASS'

datadict = dict()
for index, df in df_err_all_lowpass_subselect.iterrows():
    midweekstamp=index.strftime("%V")
    metadata='_'+index.strftime("%Y-%m-%d")
    df.index = df.index.astype('int64')    
    df_merged=df_meta.join(df)
    df_merged.drop('stationname',axis=1, inplace=True)
    df_merged.columns=['lat','lon','Node','mean']
    df_merged.dropna(inplace=True) # Cannot pass Nans to the kriging system
    df_merged.index.name = None
    #outfilename='_'.join(['stationSummaryLowpassWeekly',midweekstamp])+'.csv'
    outfilename=utilities.writeCsv(df_merged,rootdir=rootdir,subdir=subdir,fileroot='_'.join(['stationSummaryLowpassWeek',midweekstamp]),iometadata=metadata)
    datadict[midweekstamp]=outfilename
    df_merged.to_csv(outfilename)

outfilesjson = utilities.writeDictToJson(datadict, rootdir=rootdir,subdir=subdir,fileroot='runProps',iometadata='') # Never change fname
utilities.log.info('Wrote pipeline Dict data to {}'.format(outfilesjson))

print('Finished generating lowpass data files')
