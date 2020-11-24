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
from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter

def butter_lowpass_filter(df_data,filterOrder=10, numHours=200):
    numHoursCutoff=numHours
    cutoff = 2/numHoursCutoff #Includes nyquist adjustment
    #ddata=df[station]
    ddata = df_data
    print('cutoff={} Equiv number of hours is {}'.format(cutoff, 2/cutoff)) 
    filterOrder=10
    print('filterOrder is {}'.format(filterOrder))
    sos = signal.butter(filterOrder, cutoff, 'lp', output='sos',analog=False)
    datalow = signal.sosfilt(sos, ddata)
    #print('ddata {}'.format(ddata))
    #print('LP data {}'.format(datalow))
    return datalow

#
#
# Various plating routiunes
#

def makeLowpassPlot(start, end, filterOrder, lowpassAllstations):
    """
    An entry the dict (lowpassAllStations) carries all ther OBS,DIFF,LP data sets
    Plot the OBS and Diff data in a prominant way. Then layer on the cutoffs
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
    cutoffs=df_lp.columns.to_list()
    shiftDown=0.0
    for cutoff in cutoffs:
        shiftDown+=plotterDownmove
        ax.plot(df_lp[cutoff][start:end]-shiftDown,
        marker='x', markersize=2, linewidth=0.1,label='_'.join(['Lowpass',cutoff]))
    #
    ax.set_ylabel('WL (m) versus MSL')
    ax.set_title(stationName+'. Polynoimial Order='+str(filterOrder), fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(station)+'_lowpass_'+str(filterOrder)+'.png')
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
    #plt.savefig(str(station)+'_detided.png')
    plt.show()

def makeLPPlot(start, end, station, stationName, df1, df2):
    plt.close()
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
    fig, ax = plt.subplots()
    ax.plot(df1.loc[start:end],
    marker='.', markersize=1, linewidth=0.1,color='gray',label='Hourly')
    ax.plot(df2.loc[start:end],
    marker='o', color='green', markersize=1, linestyle='-',linewidth=0.5, label='Hourly-Lowpass')
    ax.set_ylabel('WL (m) versus MSL')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    #plt.savefig(str(station)+'_lowpass.png')
    plt.show()

timein = '2018-01-01 00:00'
timeout = '2018-12-31 18:00'
iometadata = ('_'+timein+'_'+timeout).replace(' ','_')

config = utilities.load_config() # Defaults to main.yml as sapecified in the config
rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

# Get HOUR  data. 3 stations are assembled in the obs.yml

#yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
yamlname='/home/jtilson/ADCIRCSupportTools/TideSIMULATION/obs.yml'

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

plt.close()
df_diff.hist()
plt.show()

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
for station in newstationlistOBS:
    print('Process station {}'.format(station))
    stationName = df_stationData.loc[int(station)]['stationname']
    lowpassdata = dict() # Carry all stations in the order processed buty first add the OBS and detided
    lowpassdata['OBS']=df_hourlyOBS[station] # Data to interpret
    lowpassdata['DETIDE']=df_diff[station] # Actual detided data set
    df_lowpass=pd.DataFrame()
    for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
        print('Process cutoff {} for station {}'.format(cutoff,station))
        df_lowpass[str(cutoff)]=butter_lowpass_filter(df_hourlyOBS[station],filterOrder=10, numHours=cutoffflank)
    df_lowpass.index = df_hourlyOBS[station].index
    lowpassdata['LP']=df_lowpass
    lowpassAllstations[station]=lowpassdata
    lowpassAllstations['station']=station
    lowpassAllstations['stationName']=stationName
    # For each station plot. OBS,explicit detided, cutoffs
    makeLowpassPlot(timein, timeout, filterOrder, lowpassAllstations)
