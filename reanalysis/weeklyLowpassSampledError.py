#!/usr/bin/env python
##
## Grab the META data JSON generated from GetObsStations: (meta json)
## Grab the Merged ADC,OBS,ERR time series data JSON computed by compError
##

import os,sys
import numpy as np
import pandas as pd
import time as tm
import matplotlib
matplotlib.use('Agg')
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

# The means are proper means for the month./week. The offset simply changes the index underwhich it will be stored
# The weekly means index at starting week +3 days.
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

# Brian prefers to use a lowpass filter than the means

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

def makePlot(start, end, station, src, stationName, dfs, dfs_7d, dfs_weekly_mean, dfs_monthly_mean, odir): 
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
    fname='/'.join([odir,station+'.png'])
    #fname=station+'.png'
    utilities.log.info('Saving station png {}'.format(fname))
    plt.savefig(fname)
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

# OrderedDicts ?
def fetch_data_metadata(f, meta):
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
    return dataDict, metaDict

def main(args):
    t0 = tm.time()

    if not args.inDir:
        utilities.log.error('Need inDir on command line: --inDir <inDir>')
        return 1
    topdir = args.inDir.strip()

    if not args.outroot:
        utilities.log.error('Need outroot on command line: --inDir <inDir>')
        return 1
    rootdir = args.outroot.strip()
    if not args.inyear:
        utilities.log.error('Need compatible year on command line: --year <year>')
        return 1
    inyear = args.inyear.strip()
    #rootdir = '/'.join([outroot,'WEEKLY'])

    # Ensure the destination is created
    ##rootdir = utilities.fetchBasedir(rootdir,basedirExtra='')

    utilities.log.info('Yearly data (with flanks) found in {}'.format(topdir))
    utilities.log.info('Actual year to process is {}'.format(inyear))
    utilities.log.info('Specified rootdir underwhich all files will be stored. Rootdir is {}'.format(rootdir))

    #f='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/adc_obs_error_merged.json'
    #meta='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/obs_water_level_metadata.json'
    f='/'.join([topdir,'adc_obs_error_merged.json'])
    meta='/'.join([topdir,'obs_water_level_metadata.json'])

    dataDict, metaDict = fetch_data_metadata(f, meta)
    stations = list(dataDict.keys()) # For subsequent naming - are we sure order is maintained?

    # Build time ranges
    timein = '-'.join([inyear,'01','01'])
    timeout = '-'.join([inyear,'12'])

    # Metadata
    df_meta=pd.DataFrame(metaDict)
    df_meta.set_index('stationid',inplace=True)

    utilities.log.info('Selecting yearly data between {} and {}, inclusive'.format(timein, timeout))
    # Time series data. This ONLY works on compError generated jsons
    df_obs_all = dictToDataFrame(dataDict, 'OBS').loc[timein:timeout] # Get whole year inclusive
    df_adc_all = dictToDataFrame(dataDict, 'ADC').loc[timein:timeout]
    df_err_all = dictToDataFrame(dataDict, 'ERR').loc[timein:timeout]

    #start = df_err_all.index.min().strftime('%Y-%m')
    #end = df_err_all.index.max().strftime('%Y-%m')
    start = df_err_all.index.min()
    # Construct new .csv files for each start-week at a single FFT lowpass cutoff
    # FFT Lowpass each station for all time. Then, extract values for all stations every start week.

    upshift=0
    hourly_cutoffs=[48]
    cutoffs = [x + upshift for x in hourly_cutoffs]

    #intersectedStations=stations # These are from the meta data not the data sets
    intersectedStations=set(df_err_all.columns.to_list()).intersection(stations) # Compares data to metadata lists
    utilities.log.info('Number of intersected stations is {}'.format(len(intersectedStations)))

    fftAllstations=dict()
    # Perform FFT for each station over the entire time range

    df_err_all_lowpass=pd.DataFrame(index=df_err_all.index)
    for station in intersectedStations:
        try:
            print('Process weekly station {}'.format(station))
            stationName = df_meta.loc[int(station)]['stationname']
            df_fft=pd.DataFrame()
            df_low = pd.DataFrame()
            print('Process cutoff {} for weekly station {}'.format(cutoff,station))
            df_temp = df_err_all[station].dropna()
            df_low['low'] = fft_lowpass(df_temp,lowhrs=cutoff)
            df_low.index=df_temp.index
            df_fft[str(cutoff)]=df_low['low']
            df_err_all_lowpass[station]=df_fft[str(cutoff)]
        except:
            utilities.log.info('weekly FFT failed for station {}'.format(station))

    # Now pull out weekly data starting at the middle of the first week
    # Build a list of indexes from which to extract data. Start at the first midweek then increment every 168 hours
    # Start from the existing data range. Get the first index value, find its week and day, then determine middle week.

# %U week number of year, with Sunday as first day of week (00..53).
# %V ISO week number, with Monday as first day of week (01..53).
# %W week number of year, with Monday as first day of week (00..53).
# 
#    numDays = (endtime-starttime).days
#    startday=pd.date_range(starttime, periods=numDays) #.values()
#    julianMetadata = startday.strftime('%y-%j').to_list()

    # Starttime: what is our firsyt data pints and what week are we in?
    # %w Weekday as a decimal number, where 0 is Sunday and 6 is Saturday

    # Now what is the neares
    df_err_all_lowpass_subselect = df_err_all_lowpass[df_err_all_lowpass.index.strftime('%w %H:%M:%S')=='0 00:00:00'] # Grab all available startweeks
    julianMetadata = df_err_all_lowpass_subselect.index.strftime('%y-%W').to_list()

    iometa = dict()
    for index,week,date in zip(range(len(df_err_all_lowpass)),julianMetadata,df_err_all_lowpass_subselect.index):
        iometa[date]='_'.join([week,date.strftime('%Y%m%d%H')])

    # df_meta and df report stationids as diff types. Yuk.
    # Store the list of filenames into a dict for krig processing

    subdir='errorfield'
    datadict = dict()
    for index, df in df_err_all_lowpass_subselect.iterrows():
        midweekstamp=index.strftime("%V")
        #metadata='_'+index.strftime("%Y-%m-%d")
        metadata='_'+iometa[index]
        df.index = df.index.astype('int64')    
        df_merged=df_meta.join(df)
        df_merged.drop('stationname',axis=1, inplace=True)
        df_merged.columns=['lat','lon','Node','state','mean']
        df_merged.dropna(inplace=True) # Cannot pass Nans to the kriging system
        df_merged.index.name = None
        #outfilename='_'.join(['stationSummaryLowpassWeekly',midweekstamp])+'.csv'
        outfilename=utilities.writeCsv(df_merged,rootdir=rootdir,subdir=subdir,fileroot='stationSummaryAves',iometadata=metadata)
        datadict[midweekstamp]=outfilename
        df_merged.to_csv(outfilename)

    outfilesjson = utilities.writeDictToJson(datadict, rootdir=rootdir,subdir=subdir,fileroot='runProps',iometadata='') # Never change fname

# Run the plot pipeline ASSUMES rootdir has been created already
    sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird
    for station in stations:
        dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d = station_level_means(df_obs_all, df_adc_all, df_err_all, station)
        start=dfs.index.min().strftime('%Y-%m')
        end=dfs.index.max().strftime('%Y-%m')
        stationName = df_meta.loc[int(station)]['stationname']
        makePlot(start, end, station, 'ERR', stationName, dfs, dfs_7d, dfs_weekly_mean, dfs_monthly_mean, rootdir) 


    utilities.log.info('Wrote pipeline Dict data to {}'.format(outfilesjson))
    print('Finished generating weekly lowpass data files')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('--inDir', action='store', dest='inDir', default=None,
                        help='directory for yearly data')
    parser.add_argument('--outroot', action='store', dest='outroot', default=None,
                        help='directory for yearly data')
    parser.add_argument('--inyear ', action='store', dest='inyear', default=None,
                        help='year to keep from the data ( removes anyu flanking months )')
    parser.add_argument('--iometadata', action='store', dest='iometadata',default='', help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default='', help='Used to locate output files into subdir', type=str)
    args = parser.parse_args()
    sys.exit(main(args))



