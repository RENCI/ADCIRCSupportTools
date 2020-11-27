#
# Hourly data from COOPs is NOT the hourly averged OBS 6min data. 
# Then take the difference to compute a residual that depicts "truth"
# Then compare differeing degress of lowpass filtering on the OBS data to achieve
#
# NOTE brian suggested grabbing hourly data from COOPS in stead of 6min, but I want to test like this first before
# changing the AST codes since our ave 6min should be the same as COOPS.

import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter

print('pandas Version={}'.format(pd.__version__))
print('seaborn Version={}'.format(sns.__version__))
print('scipy Version={}'.format(scipy.__version__))
print('numpy Version={}'.format(np.__version__))

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

#
# FFT lowpass provided via B.Blanton
#
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

#
# Various plating routines
#
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
    ax.plot(lowpassAllstations[station]['DETIDE'][start:end],
    color='black', marker='o',markersize=2, linewidth=.5, linestyle='-', label='Detided Hourly')
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

def makePlot(start, end, station, stationName, df1, df2, df3):
    plt.close()
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
    fig, ax = plt.subplots()
    ax.plot(df1.loc[start:end],
    marker='.', markersize=1, linewidth=0.1,color='gray',label='Hourly')
    ax.plot(df2.loc[start:end],
    color='red', alpha=0.3, linewidth=.5, linestyle='-', label='Predicted')
    ax.plot(df3.loc[start:end],
    marker='o', color='green', markersize=1, linestyle='-',linewidth=0.5, label='Hourly-Predicted')
    ax.set_ylabel('WL (m) versus MSL')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(station)+'_detided.png')
    #plt.show()

def makeLowpassFOPlot(start, end, cutoff, lowpassAllstations):
    """
    An entry the dict (lowpassAllStations) carries all the OBS,DIFF,LP data sets
    Plot the OBS and Diff data in a prominant way. Then layer on the diff filterOrders
    """
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
    ax.plot(lowpassAllstations[station]['DETIDE'][start:end],
    color='black', marker='o',markersize=2, linewidth=.5, linestyle='-', label='Detided Hourly')
    # Now start with the lowpass filters
    df_lp = lowpassAllstations[station]['LP']
    filterOrders=df_lp.columns.to_list()
    shiftDown=0.0
    for filterOrder in filterOrders:
        shiftDown+=plotterDownmove
        ax.plot(df_lp[filterOrder][start:end]-shiftDown,
        marker='x', markersize=2, linewidth=0.1,label='='.join(['LP. Order',filterOrder]))
    #
    ax.set_ylabel('WL (m) versus MSL')
    ax.set_title(stationName+'. Cutoff='+str(cutoff), fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(station)+'_lowpass_filterOrdersweep_'+str(cutoff)+'.png')
    #plt.show()

def makeLowpassHist(start, end, lowpassAllstations, filterOrder='', metadata=['lowpass_histo','LP']):
    """
    An entry the dict (lowpassAllStations) carries all the OBS,DIFF,LP data sets
    Plot the OBS and Diff data in a prominant way. Then layer on the cutoffs
    """
    colMetadata = metadata[1]
    nameMetadata = metadata[0]
    station=lowpassAllstations['station']
    stationName=lowpassAllstations['stationName']
    print('station {} Name {}'.format(station, stationName))
    #
    plt.close()
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
    fig, ax = plt.subplots()
    # OBS and DIFF
    #df_temp=lowpassAllstations[station]['OBS'][start:end]
    #df_temp.columns='OBS'
    #df_temp.plot(kind='hist',alpha=1.0,color='gray',bins=40)
    df_temp=lowpassAllstations[station]['DETIDE'][start:end]
    df_temp.columns='DETIDE'
    df_temp.plot(kind='hist',alpha=1.0,color='black',bins=40)
    # Now start with the lowpass filters
    df_lp = lowpassAllstations[station][colMetadata]
    cutoffs=df_lp.columns.to_list()
    for cutoff in cutoffs:
        df_temp=df_lp[cutoff][start:end]
        df_temp.columns='LP-'+str(cutoff)+'hr'
        df_temp.plot(kind='hist',alpha=0.4,bins=40)
    #
    ax.set_ylabel('WL (m) versus MSL')
    if filterOrder=='':
        ax.set_title(stationName+'. FFT', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    else:
        ax.set_title(stationName+'. Polynomial Order='+str(filterOrder), fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fileMetaname = '_'.join([str(station),nameMetadata])
    fileName = fileMetaname+'.png' if filterOrder=='' else fileMetaname+'_'+str(filterOrder)+'.png'
    plt.savefig(fileName)
    #plt.show()
##
## Do some work
##

timein = '2018-01-01 00:00'
timeout = '2018-12-31 18:00'
#timeout = '2018-01-31 18:00'

##timein = '2018-09-01 00:00'
##timeout = '2018-10-01 18:00'
iometadata = ('_'+timein+'_'+timeout).replace(' ','_')

plot_timein='2018-09-01 00:00'
plot_timeout='2018-09-30 18:00'

config = utilities.load_config() # Defaults to main.yml as sapecified in the config
rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

# Get HOUR  data. 3 stations are assembled in the obs.yml

#yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
yamlname='/home/jtilson/ADCIRCSupportTools/TideSIMULATION/obs.yml'
#yamlname='/home/jtilson/ADCIRCSupportTools/config/obs.yml'

rpl = GetObsStations(product='hourly_height', rootdir=rootdir, yamlname=yamlname, metadata=iometadata)
stations = rpl.stationListFromYaml()
df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)
##df_stationNodelist = rpl.fetchStationNodeList()
##stations = df_stationNodelist['stationid'].to_list()
##df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)
df_hourlyOBS, count_nanOBS, newstationlistOBS, excludelistOBS = rpl.fetchStationProductFromIDlist(timein, timeout,interval='h')
retained_times = df_hourlyOBS.index.to_list() # some may have gotten wacked during the smoothing`
#listSuspectStations = rpl.writeURLsForStationPlotting(newstationlistOBS, timein, timeout)
#detailedpklOBS, smoothedpklOBS, metapklOBS, urlcsvOBS, exccsvOBS, metaJOBS, detailedJOBS, smoothedJOBS = rpl.fetchOutputNames()


# Get PRE data. 3 stations are assembled in the obs.yml
rpl2 = GetObsStations(product='predictions', rootdir=rootdir, yamlname=yamlname, metadata=iometadata)
stations = rpl2.stationListFromYaml()
df_stationData, stationNodelist = rpl2.fetchStationMetaDataFromIDs(stations)
##df_stationNodelist = rpl2.fetchStationNodeList()
##stations = df_stationNodelist['stationid'].to_list()
##df_stationData, stationNodelist = rpl2.fetchStationMetaDataFromIDs(stations)
df_hourlyPRE, count_nanPRE, newstationlistPRE, excludelistPRE = rpl2.fetchStationProductFromIDlist(timein, timeout,interval='h')
retained_times = df_hourlyPRE.index.to_list() # some may have gotten wacked during the smoothing`
#listSuspectStationsPRE = rpl2.writeURLsForStationPlotting(newstationlistPRE, timein, timeout)
#detailedpklPRE, smoothedpklPRE, metapklPRE, urlcsvPRE, exccsvPRE, metaJPRE, detailedJPRE, smoothedJPRE = rpl2.fetchOutputNames()
# Get the difference
print('Compute diffs')
df_diff = df_hourlyOBS-df_hourlyPRE

# Get meta data
df_stationData.set_index('stationid',inplace=True)

intersectedStations = list(set(newstationlistPRE) & set(newstationlistOBS))
print('Number of intersected stations {}'.format(len(intersectedStations)))

#for station in intersectedStations:
#    print('Plot station data {}'.format(station))
#    stationName = df_stationData.loc[int(station)]['stationname']
#    makePlot(timein, timeout, station, stationName, df_hourlyOBS[station], df_hourlyPRE[station], df_diff[station])

#plt.close()
#df_diff.hist()
#plt.show()

# Perform a sweep of cutoffs (in hours) to test sensitivity to that
# Nominally 1hr,24hr,48hr,1wk, 1month] plus 4 extra hours

upshift=4
#hourly_cutoffs=[12,24,48,168,720]
hourly_cutoffs=[12,24,48,168]
cutoffs = [x + upshift for x in hourly_cutoffs] 
# Do we need to carry colors ?
# Create a station df with columns of cutoffs

filterOrder=10
lowpassAllstations=dict()
for station in intersectedStations:
    print('Process station {}'.format(station))
    stationName = df_stationData.loc[int(station)]['stationname']
    lowpassdata = dict() # Carry all stations in the order processed buty first add the OBS and detided
    lowpassdata['OBS']=df_hourlyOBS[station] # Data to interpret
    lowpassdata['DETIDE']=df_diff[station] # Actual detided data set
    df_lowpass=pd.DataFrame()
    for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
        print('Process cutoff {} for station {}'.format(cutoff,station))
        df_temp = df_hourlyOBS[station].dropna()
        df_lowpass[str(cutoff)]=butter_lowpass_filter(df_temp,filterOrder=10, numHours=cutoffflank)
    df_lowpass.index = df_temp.index
    lowpassdata['LP']=df_lowpass
    lowpassAllstations[station]=lowpassdata
    lowpassAllstations['station']=station
    lowpassAllstations['stationName']=stationName
    # For each station plot. OBS,explicit detided, cutoffs
    makeLowpassPlot(plot_timein, plot_timeout, lowpassAllstations, filterOrder=filterOrder)
    makeLowpassHist(plot_timein, plot_timeout, lowpassAllstations, filterOrder=filterOrder)

print('Start filterOrder sweep')
filterOrders=[1,2,3,4,5,6,50,7,8,9,10]

lowpassFOAllstations=dict()
cutoff=24+4 # 24 hours plus a 4 hour flank
hourly_cutoff=24
for station in intersectedStations:
    print('Process station {}'.format(station))
    stationName = df_stationData.loc[int(station)]['stationname']
    lowpassFOdata = dict() # Carry all stations in the order processed buty first add the OBS and detided
    lowpassFOdata['OBS']=df_hourlyOBS[station] # Data to interpret
    lowpassFOdata['DETIDE']=df_diff[station] # Actual detided data set
    df_lowpass=pd.DataFrame()
    for filterOrder in filterOrders:
        print('Process cutoff {} for station {} filterOrder {}'.format(cutoff,station,filterOrder))
        df_temp = df_hourlyOBS[station].dropna()
        df_lowpass[str(filterOrder)]=butter_lowpass_filter(df_temp,filterOrder=filterOrder, numHours=cutoff)
    df_lowpass.index = df_temp.index
    lowpassFOdata['LP']=df_lowpass
    lowpassFOAllstations[station]=lowpassFOdata
    lowpassFOAllstations['station']=station
    lowpassFOAllstations['stationName']=stationName
    # For each station plot. OBS,explicit detided, cutoffs
    makeLowpassFOPlot(plot_timein, plot_timeout, hourly_cutoff, lowpassFOAllstations)

#
# Try the FFT lowpass
#
upshift=4
#hourly_cutoffs=[12,24,48,168,720]
hourly_cutoffs=[12,24,48,168]
cutoffs = [x + upshift for x in hourly_cutoffs]

fftAllstations=dict()
for station in intersectedStations:
    print('Process station {}'.format(station))
    stationName = df_stationData.loc[int(station)]['stationname']
    fftdata = dict() # Carry all stations in the order processed buty first add the OBS and detided
    fftdata['OBS']=df_hourlyOBS[station] # Data to interpret
    fftdata['DETIDE']=df_diff[station] # Actual detided data set
    df_fft=pd.DataFrame()
    for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
        print('Process cutoff {} for station {}'.format(cutoff,station))
        df_temp = df_hourlyOBS[station].dropna()
        df_fft[str(cutoff)]=fft_lowpass(df_temp,lowhrs=cutoffflank)
    df_fft.index = df_temp.index
    fftdata['FFT']=df_fft
    fftAllstations[station]=fftdata
    fftAllstations['station']=station
    fftAllstations['stationName']=stationName
    # For each station plot. OBS,explicit detided, cutoffs
    makeLowpassPlot(plot_timein, plot_timeout, fftAllstations, filterOrder='', metadata=['lowpass_fft','FFT'])
    makeLowpassHist(plot_timein, plot_timeout, fftAllstations, filterOrder='', metadata=['lowpass_fft_histo','FFT'])

#station=8658163
#cutoff=28 # 24 hours plus a 4 hour flank
#lowhrs=cutoff
#df_temp=df_hourlyOBS[station].dropna()
#datafft=fft_lowpass(df_temp,lowhrs)
#df_temp['FFT']=datafft
