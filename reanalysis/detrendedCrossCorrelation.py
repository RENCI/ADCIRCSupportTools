#!/usr/bin/env python
##
## Here we perform cross-correlations testing between stations using the OBS WL data. 
## We want to remove diurnal behavior as, otherwise that would dominate the correaltion
## Use a 7 day lowpass FFT (and possible others) prior to the correlation test. Then, go
## and mean=0,var=1 standardize each of the time series.
## Lastly compute the cross correlations 
##
## Some of the year data includes flanking, so remove them 
##

import os,sys
import numpy as np
import pandas as pd
import time as tm
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas.tseries.frequencies import to_offset
import json
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, savgol_filter
from sklearn.preprocessing import StandardScaler
import datetime as dt
from utilities.utilities import utilities
from scipy.signal import correlate
from scipy.sparse.csgraph import reverse_cuthill_mckee,\
        maximum_bipartite_matching
from scipy.sparse import diags, csr_matrix, coo_matrix

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os,sys,glob

# Ordinarily, we would pass the regular data in. DO we want to passd in correlations? 

def runPCA(indf,keepCols):
    nComponents = 30
    df = scaledf(indf,keepCols)
    print('execute new PCA')
    pca = PCA(n_components=nComponents)
    pca_result = pca.fit_transform(df.drop(keepCols,axis=1).values)
    H = pca_result
    df_H=pd.DataFrame(H)
    df_H.index = df.drop(keepCols,axis=1).index
    return pd.merge(df.loc[:,keepCols],df_H,left_index=True,right_index=True)

def runTSNE(df, keepCols=None): # Can add in STATE or other identyfiers from metadata
    print('execute tSNE')
    perplex=10
    time_start = tm.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplex, n_iter=1000)
    print('fit tSNE')
    tsne_results = tsne.fit_transform(df.drop(keepCols,axis=1).values)
    #tsne_results = tsne.fit_transform(df.values)
    print ('t-SNE done! Time elapsed: {} seconds '.format(tm.time()-time_start))
    df_subset = pd.DataFrame([tsne_results[:,0],tsne_results[:,1]]).T
    df_subset.columns=("tsne-2d-one","tsne-2d-two")
    df_subset.index = df.index
    #
    keepCells=df_subset.index
    df_plotData = pd.merge(df.loc[keepCells,keepCols],df_subset.loc[keepCells],left_index=True,right_index=True)
    #df_plotData = pd.merge(df,df_subset,left_index=True,right_index=True)
    #df_plotData = df_subset
    return df_plotData

def genSinglePlot(i, fig, df_plotDatain,effect,numeffect,inputMetadata):
    sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird
    #df_plotData = df_plotDatain.sample(frac=0.7, random_state=1)
    df_plotData = df_plotDatain
    ax = fig.add_subplot(1,1,i)
    #markers= {'ME':'o', 'MA':'v', 'CT':'^', 'NY':'<', 'NJ':'>', 'DE':'8','MD':'s','VA': 'p','NC': '*', 'SC':'h', 'GA':'H', 'FL':'D', 'AL':'d','MS': 'P','LA': 'X', 'TX':'o'}
    markers= ['o', 'v', '^', '<', '>', '8','s','p','*','h','H','D','d','P','X', 'o']
    plt.scatter(
        x='tsne-2d-one', y='tsne-2d-two',
        #hue=effect, # 
        #style=effect,
        #palette=sns.color_palette("husl",numeffect),
        #palette=sns.color_palette("hls", numeffect),
        data=df_plotData,
        #legend=False,
        colors=markers
        #s=500, # 1000,
        c='state'
        #sizes=(20, 200),
        #edgecolor="none",
        #marker=markers,
        #alpha=.8
        )
    #ax.set_title(inputMetadata, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_ylabel('')
    ax.set_xlabel('')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    #plt.tight_layout()
    plt.axis('tight')

def genSingleLMPLOTPlot(i, fig, df_plotDatain,effect,numeffect,inputMetadata):
    sns.set(rc={'figure.figsize':(11, 4)}) # Setr gray background and white gird

    #df_plotData = df_plotDatain.sample(frac=0.7, random_state=1)
    df_plotData = df_plotDatain
    ax = fig.add_subplot(1,1,i)
    #ax = plt.subplots()
    #markers = df_plotData[effect]
    #markers = df_plotData[effect].astype('catagory').cat.codes
    markers= ['o', 'v', '^', '<', '>', '8','s','p','*','h','H','D','d','P','X', 'o']

    #markers= {'ME':'o', 'MA':'v', 'CT':'^', 'NY':'<', 'NJ':'>', 'DE':'8','MD':'s','VA': 'p','NC': '*', 'SC':'h', 'GA':'H', 'FL':'D', 'AL':'d','MS': 'P','LA': 'X', 'TX':'o'}
    sns.lmplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue=effect, # 
        #style=effect,
        palette=sns.color_palette("husl",numeffect),
        #palette=sns.color_palette("hls", numeffect),
        data=df_plotData,
        #legend=False,
        #s=1000,
        #sizes=(20, 200),
        #edgecolor="none",
        markers=markers,
        #alpha=.8
        )
    #ax.set_title(inputMetadata, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_ylabel('')
    ax.set_xlabel('')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    #plt.tight_layout()
    plt.axis('tight')



#############################################################################

def intersectTimes(x,y):
    """
    Sometimes a NaN occurs in sttion data. Correlate cannot handle neither:
    Nans nor vector of differing lengths. Thanks.
    """
    inter = x.index & y.index
    return x.loc[inter], y.loc[inter]

def computeRxyafterLagging(x,y):
    """Compute the crosscoerrelation lagging then shift the Y data set
       compute the single Rxy on the best mapped Y with X 
       np.roll
    """
    lag = np.argmax(correlate(x, y))   # returns array index of max position. we want +1 for rolling
    #print('lag is '+str(lag))
    y_clag = np.roll(y, shift=int(np.ceil(lag+1)))
    Rxy = np.corrcoef(x,y_clag).item(2)
    return (Rxy,lag)

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
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
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
    df_temp=lowpassAllstations[station]['OBS'][start:end]
    df_temp.columns='OBS'
    df_temp.plot(kind='hist',alpha=1.0,color='gray',bins=40)
    #df_temp=lowpassAllstations[station]['ERR'][start:end]
    #df_temp.columns='ER'
    #df_temp.plot(kind='hist',alpha=1.0,color='black',bins=40)
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
    #ax.plot(lowpassAllstations[station]['ERR'][start:end],
    #color='black', marker='o',markersize=2, linewidth=.5, linestyle='-', label='ADCIRC-OBS')
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
    print('SAVE {}'.format(fileName))
    plt.savefig(fileName)
    #plt.show()

def RCMDataFrame(df):
    """ Take the corvaiance matrix and reorder assuming sparsity using RCM
    Return the 2D matrix as a np_array. We need to convert the COV matrix into
    0/1 format based on some threshold. Easy just convert to a boolean
    """
    threshold = 0.80 # presumes (0,1) values as these have already been scaled
    cov = df.cov().values
    #adj = (cov > threshold).astype(int)
    #graph = coo_matrix(adj).tocsr()
    #perm = reverse_cuthill_mckee(graph, None).tolist()
    #N = cov.shape[1]
    #for i in range(N):
    #    cov[:,i]= cov[perm,i]
    #for i in range(N):
    #    cov[i,:]= cov[i,perm]
    #newDF = pd.DataFrame(cov)
    #Ereturn (newDF)

def covariance_matrix_rcm(df,titletext):
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    df_rcm = RCMDataFrame(df)
    df_rcm.to_pickle('RCM.png')
    cax = ax1.imshow(df_rcm, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title(titletext)
    labels=[ df.columns.values ]
    ####ax1.set_xticklabels(labels,fontsize=6)
    #####ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
#    plt.filename('covarianceMatrix_scaled_unwhitened_80.pdf')
    plt.show()


#
# Start the work
#
def main(args):
    t0 = tm.time()

    inyear='2018'
    inDir='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/reanalysis/TESTFULL/STATE/YEARLY-2018/KRIG_LONGRANGE'
    #inDir='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/reanalysis/TESTFULL/YEARLY-2018/KRIG_CV_LONGRANGE'

    outroot='.'
    rootdir=outroot
    topdir=inDir

    if not args.inDir:
        utilities.log.error('Need inDir on command line: --inDir <inDir>')
    #    return 1
    #topdir = args.inDir.strip()
    if not args.outroot:
        utilities.log.error('Need outroot on command line: --inDir <inDir>')
    #    return 1
    #rootdir = args.outroot.strip()
    if not args.inyear:
        utilities.log.error('Need compatible year on command line: --year <year>')
    #    return 1
    #inyear = args.inyear.strip()

    # Ensure the destination is created
    ##rootdir = utilities.fetchBasedir(rootdir,basedirExtra='')

    utilities.log.info('Yearly data (with flanks) found in {}'.format(topdir))
    utilities.log.info('Actual year to process is {}'.format(inyear))
    utilities.log.info('Specified rootdir underwhich all files will be stored. Rootdir is {}'.format(rootdir))

    #f='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/adc_obs_error_merged.json'
    #meta='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/2018-Reanalysis-ERR_weeklyMeans/JAN72018/obs_water_level_metadata.json'
    #topdir='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/reanalysis/TESTFULL/YEARLY-2018/KRIG_CV_LONGRANGE'

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

    print('META {}'.format(df_meta))

    utilities.log.info('Selecting yearly data between {} and {}, inclusive'.format(timein, timeout))
    # Time series data. This ONLY works on compError generated jsons
    df_obs_all = dictToDataFrame(dataDict, 'OBS').loc[timein:timeout] # Get whole year inclusive
    df_adc_all = dictToDataFrame(dataDict, 'ADC').loc[timein:timeout]
    df_err_all = dictToDataFrame(dataDict, 'ERR').loc[timein:timeout]

    df_data = df_obs_all
    df_diff=df_err_all

    ##############################################################################################
    # Construct new .csv files for each mid-week at a single FFT lowpass cutoff
    # FFT Lowpass each station for all time. Then, extract values for all stations every mid week.

    upshift=4
    hourly_cutoffs=[12,24,48,168]
    cutoffs = [x + upshift for x in hourly_cutoffs]
    intersectedStations=stations
    fftAllstations=dict()

    plot_timein = timein
    plot_timeout = timeout

    # Perform FFT for each station observations over the entire time range
    #df_data_lowpass=pd.DataFrame()
    for station in intersectedStations:
        print('Process station {}'.format(station))
        stationName = df_meta.loc[int(station)]['stationname']
        fftdata = dict()
        fftdata['OBS']=df_obs_all[station] # Data to interpret
        fftdata['ERR']=df_diff[station] # Actual detided data set
        df_fft=pd.DataFrame()
        for cutoffflank,cutoff in zip(cutoffs,hourly_cutoffs):
            print('Process cutoff {} for station {}'.format(cutoff,station))
            df_temp = df_data[station].dropna()
            df_fft[str(cutoff)]=fft_lowpass(df_temp,lowhrs=cutoffflank)
        df_fft.index = df_temp.index
        fftdata['FFT']=df_fft
        fftAllstations[station]=fftdata
        fftAllstations['station']=station
        fftAllstations['stationName']=stationName
        makeLowpassPlot(plot_timein, plot_timeout, fftAllstations, filterOrder='', metadata=['lowpass_fft','FFT'])
        makeLowpassHist(plot_timein, plot_timeout, fftAllstations, filterOrder='', metadata=['lowpass_fft_histo','FFT'])

    # Choose the FFT pas threshold for which the correlations will be determined

    # Grab the FFT data. It is in the form of a dict['stations']
    df_lowpass = pd.DataFrame()
    for station in stations:
        df_lowpass[station]= fftAllstations[station]['FFT']['168']

    # Another opportunity to restrict the time range
    starttime=''.join([inyear,'-01-01 00:00:00'])
    endtime=''.join([inyear,'-12-31 18:00:00'])
    df_data_lowpass_subselect = df_lowpass[starttime:endtime]

    # Now standardize each station using mean=0,var=1 
    scaler = StandardScaler()
    data_lowpass_scaled = scaler.fit_transform(df_data_lowpass_subselect.values)
    df_data_lowpass_scaled=pd.DataFrame(data_lowpass_scaled, index=df_data_lowpass_subselect.index, columns=df_data_lowpass_subselect.columns)
    df_data_lowpass_scaled.describe() # All mean=0, std=1
    df_data_lowpass_scaled.to_pickle('ScaledOberservationData.pkl')

    # Simple correlation heatmap
    plt.close()
    sns.set(rc={'figure.figsize':(11, 6)})
    sns.heatmap(df_data_lowpass_scaled.corr())
    plt.savefig('heatmap_scaled_nolagging.png')

# Run the averaging plot pipeline ASSUMES rootdir has been created already
    sns.set(rc={'figure.figsize':(11, 4)}) # Set gray background and white gird
    for station in stations:
        dfs, dfs_weekly_mean, dfs_monthly_mean, dfs_7d = station_level_means(df_obs_all, df_adc_all, df_err_all, station)
        start=dfs.index.min().strftime('%Y-%m')
        end=dfs.index.max().strftime('%Y-%m')
        stationName = df_meta.loc[int(station)]['stationname']
        makePlot(start, end, station, 'OBS', stationName, dfs, dfs_7d, dfs_weekly_mean, dfs_monthly_mean, rootdir) 

    df_state=df_meta['state']
    print('State Meta {}'.format(df_state))

    # Build a corr-matrix using optimum lagging (may not matter)
    df_corr_matrix=pd.DataFrame()
    lagging = list() 
    for station in stations: 
        x=df_data_lowpass_scaled[station].dropna()
        rcorrs=list()
        for s in stations: 
            y = df_data_lowpass_scaled[s].dropna()
            xnew,ynew=intersectTimes(x,y)
            rcorr, lag = computeRxyafterLagging(xnew,ynew)
            lagging.append(lag)
            rcorrs.append(rcorr)
        df_corr_matrix[station]=rcorrs
    df_corr_matrix.index = df_data_lowpass_scaled.columns
    df_corr_matrix.columns = df_data_lowpass_scaled.columns

    print('PLOTTER')
    plt.close()
    plt.plot(lagging)
    plt.savefig('lagging.png')
    plt.show()

    # The lagged corr is not much different than the original .corr() data
    plt.close()
    sns.set(rc={'figure.figsize':(11, 6)})
    sns.heatmap(df_corr_matrix)
    plt.savefig('heatmap_scaled_lagging.png')

    # Now can we RCM cluster the data 
    #covariance_matrix_rcm(df_corr_matrix,'titletext')
    # This can be used to identyify stratified sampling for the CV kriging.
    threshold = 0.70 # presumes (0,1) values as these have already been scaled
    cov = df_corr_matrix.values
    #adj = (cov > threshold).astype(int)
    adj = cov
    graph = coo_matrix(adj).tocsr()
    perm = reverse_cuthill_mckee(graph, None)
    # Return perm the reordered set of indices. Now need to reorder df_corr_matrix

    df_test = df_corr_matrix.iloc[perm,perm] # Reorder the row and columns
    plt.close()
    sns.set(rc={'figure.figsize':(11, 6)})
    sns.heatmap(df_test)
    plt.savefig('heatmap_rcm.png')

    df_adj = (df_test > threshold).astype(int)
    plt.close()
    sns.set(rc={'figure.figsize':(11, 6)})
    sns.heatmap(df_adj)
    plt.savefig('heatmap_adj.png')

    # Lastly build a new object that associates those that 
    # TSNE or UMAP cluster the data into groups to identify 
    # new groupIDs for stratified cluster CVs.

    df_corr_matrix.to_pickle('df_corr_matrix.pkl')
    df_state.to_pickle('df_state.pkl')

    df_state.index = df_state.index.astype('str') # Crud stationID are ints for meta, strings for data
    df_corr_matrix = df_corr_matrix.join(df_state)

    #df_plotData = pd.merge(df.loc[keepCells,keepCols],df_subset.loc[keepCells],left_index=True,right_index=True)

    print('MERGED {}'.format(df_corr_matrix))
    df_plotData = runTSNE(df_corr_matrix, keepCols=['state'])
    print('PLOT {}'.format(df_plotData))
    
    numeffect = 16
    i=1
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    genSinglePlot(i, fig, df_plotData,'state',numeffect,'inputMeta')
    plt.savefig('cluster.png')
    #plt.show()

    ###perm = reverse_cuthill_mckee(df_corr_matrix)
    #arr_perm = arr_orig[perm, perm]

    #import networkx as nx
    #from networkx.utils import cuthill_mckee_ordering
    #G = nx.gnm_random_graph(n=30, m=55, seed=1)
    #rcm = list(cuthill_mckee_ordering(G))
    #d = {node:rcm.index(node) for node in G}
    #H = nx.relabel_nodes(G, mapping=d)
    #H_adj = nx.adjacency_matrix(H,nodelist=range(30))
    #plt.spy(H_adj)
    #plt.show()

    # Maybe try to cluster using the original scaled data sets? (station x times)
    #df_scaled = df_data_lowpass_scaled.T
    #df_corr = df_scaled.join(df_state)
    #df_plotData_scale = runTSNE(df_corr, keepCols=['state'])
    #numeffect = 16
    #i=1
    #fig = plt.figure(figsize=(10, 8))
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #genSinglePlot(i, fig, df_plotData_scale,'state',numeffect,'inputMeta')
    #plt.show()




    


    print('Finished generating Correlation')

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



