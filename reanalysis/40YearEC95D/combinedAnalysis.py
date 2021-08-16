#!/usr/bin/env python
##
## Grab the META data JSON generated from GetObsStations: (meta json)
## Grab the Merged ADC,OBS,ERR time series data JSON computed by compError
##
## NOTE The sa

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
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Prepare for different kinds of plots
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

def fetchDataFrameSource(indf, name='ERR'):
    """
    Munge through the data frame keeping only the named rows
    Then push TIME to the index and ensure they are pd.Timestamps
    """
    df=indf[indf.SRC==name].drop('SRC',axis=1)
    df.set_index('TIME',inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def fetch_metadata(injson):
    """
    Read the current year's metadata. The number of stations per metadata may change
    But For our use cases they do not change
    """
    with open(injson, 'r') as fp1:
        try:
            dataDict = json.load(fp1)
        except OSError:
            print("Could not open/read file {}".format(injson))
            sys.exit()
    df_meta = pd.DataFrame(dataDict)
    return df_meta.set_index('stationid') 
    
def noFitStationPlot(station, stationName, df_ADC, df_OBS, df_ERR):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(font_scale=1.0, style="darkgrid")
    ax.plot(df_OBS,
        marker='.', markersize=1, linestyle='-',linewidth=0.1,color='darkgray', alpha=0.3,label='OBS')
    ax.plot(df_ADC,
        marker='.', markersize=0, linestyle='-', linewidth=0.1,color='lightblue',label='ADC')
    ax.plot(df_ERR,
        marker='.', markersize=.0, linestyle='-', linewidth=0.1,color='red',label='ADC-OBS')
    ax.set_ylabel('meters')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); #-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    #plt.savefig(str(station)+'.png')
    plt.show()

def fitStationPlot(station, stationName, df_ADC, df_OBS, df_ERR):
    col = sns.color_palette('bright')[0:4] 
    print('cols {}'.format(col))
    df = df_ERR.copy()
    df['hours_from_start'] = (df.index - df.index[0]).total_seconds()//3600
    x_all = df['hours_from_start'].values.reshape(-1, 1)
    y_all =df[str(station)].values
    # Cannot fit using nans
    df = df.dropna()
    x = df['hours_from_start'].values.reshape(-1, 1)
    y = df[str(station)].values
    model = linear_model.LinearRegression().fit(x, y)
    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    # Scale
    intercept=model.intercept_ #intercept units in m is okay
    slope=(model.coef_)*24*365*1000 #convert slope to mm/yr from m/hr
    intercept=round(intercept,2) # Keep in meters
    slope=round(slope[0],0)
    # Compare get some basic stats
    ypred = model.predict(x_all)
    # Prepare data for overlay plotting
    df_FIT=df_ERR.copy()
    df_FIT['FIT']=ypred
    # Form plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(font_scale=1.0, style="darkgrid")
    col = sns.color_palette("bright")[0:3]

    # Alpha channels are not helping sometimes ADS > OBS and sometimes vice versa
    # Try to detemrtine which is larger
    if max(df_OBS.values) > max(df_ADC.values):
        ax.plot(df_OBS,
            marker='.', markersize=1, linestyle='-',linewidth=0.2, color=col[0], label='NOAA/NOS')
        ax.plot(df_ADC,
            marker='.', markersize=0, linestyle='-', linewidth=0.3, color=col[1],label='ADCIRC')
    else:
        ax.plot(df_ADC,
            marker='.', markersize=0, linestyle='-', linewidth=0.3, color=col[1],label='ADCIRC')
        ax.plot(df_OBS,
            marker='.', markersize=1, linestyle='-',linewidth=0.2, color=col[0], label='NOAA/NOS')
    ax.plot(df_ERR,
        marker='.', markersize=.0, linestyle='-', linewidth=0.2,color=col[2],label='ADC-NOS')
    ax.plot(df_FIT['FIT'],
        markersize=.0, linestyle='-', linewidth=0.3,color='black',label=r"$\mathbb{E}(ADC-NOS)$")

    ax.set_ylabel('meters')
    ax.set_title(stationName, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); #-%m'));
    ax.legend(fontsize=10);
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    fitresult='Intercept %s m, slope %s mm/yr'%(intercept, slope)
    plt.text('1980', -1.0, fitresult, horizontalalignment='left', size='medium', color='black')
    plt.tight_layout()
    plt.savefig(str(station)+'_withFit.png')
    plt.close()
    #plt.show()

# Start the work 
# Read the station metadata since we will want lon/lat/stationnames later on
# The stations are already ordered in the file itself.

# List of all years for combining
years=[1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

basefilename='adc_obs_error_merged.csv'
metafilename='obs_hourly_height_metadata.json'
#basedirectory='/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/40YearEC95D/EC95D-DA'
basedirectory='/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/40YearEC95D/EC95D'

# Loop overall data sets and read data files
yearlyData=list()
yearlyMeta=list()

for year in years:
    print('Processing year {}'.format(year))
    fulldirectory=basedirectory+'/YEARLY-'+str(year)
    df = pd.read_csv('/'.join([fulldirectory,basefilename]), header=0)
    yearlyMeta.append(fetch_metadata('/'.join([fulldirectory,metafilename])))
    yearlyData.append(fetchDataFrameSource(df, name='ERR'))

# Merge all the data. expand times/stations at-will
df_all = pd.concat(yearlyData)

# Get the meta data
df_meta = pd.concat(yearlyMeta)
df_meta.drop_duplicates(subset=None, keep='first',inplace=True)
stationOrder = [str(id) for id in df_meta.index.to_list()][::-1]

# Reorder the data to match the metadata order for sensible plotting
df_all = df_all[stationOrder]

# Start statistical processing. Can we groupby year and get station averages?
# https://kanoki.org/2020/05/26/dataframe-groupby-date-and-time/
df_all.groupby(pd.Grouper(level='TIME',freq='Y')).mean()

#############################################################################3
# Distribution of yearly mean errors for each TIME. 
# Possibly shows non-stationarity with greater errors at later times.
#     But at longer times we also have more stations, so that may actually be the "trend"

# Python plot formatting is a horrible mess.

fig, ax = plt.subplots(figsize=(12, 8))
sns.set(font_scale=1.0, style="darkgrid")
df_all.groupby(pd.Grouper(level='TIME',freq='Y')).mean().T.boxplot()
plt.title('Distribution of station WL means for the indicated year: ec95d: ADC-OBS')
plt.ylabel('meters')
plt.axhline(df_all.mean().mean(), c='r')
plt.gcf().subplots_adjust(bottom=0.30)
##ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=12))
##ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); #-%m'));
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig('yearlyMeansVsTimes.png')

###############################################################################
# Swarm: Distributionm of monthly mean errors for each station. 
# Use SNS method instead
# Incredible nonsense for making black boxes
# We can color by STATE ?

fig, ax = plt.subplots(figsize=(12, 8)) 
sns.set(font_scale=1.0, style="darkgrid")
df_data = df_all.groupby(pd.Grouper(level='TIME',freq='Y')).mean().T
df_data.columns=df_data.columns.strftime('%Y')
sns.boxplot(x="TIME", y="value", data=pd.melt(df_data), color='white', width=.5, fliersize=0)
for i,box in enumerate(ax.artists): # The dumbest python / sns thing yet.
    box.set_edgecolor('black')
    box.set_facecolor('white')
        # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         ax.lines[j].set_color('black')
sns.swarmplot(x="TIME", y="value", data=pd.melt(df_data), color='lightgreen')
plt.title('Distribution of station WL means for the indicated year: ec95d: ADC-OBS')
plt.ylabel('meters')
ax.set(xlabel=None)
plt.axhline(df_all.mean().mean(), c='r')
plt.gcf().subplots_adjust(bottom=0.30)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig('yearlyMeansVsTimesSwarm.png')

###############################################################################
# Violin: Distributionm of monthly mean errors for each station. 
# Use SNS method instead
# Incredible nonsense for making black boxes

fig, ax = plt.subplots(figsize=(12, 8))
sns.set(font_scale=1.0, style="darkgrid")
df_data = df_all.groupby(pd.Grouper(level='TIME',freq='Y')).mean().T
df_data.columns=df_data.columns.strftime('%Y')
sns.boxplot(x="TIME", y="value", data=pd.melt(df_data), color='white', width=.5, fliersize=0)
for i,box in enumerate(ax.artists): # The dumbest python / sns thing yet.
    box.set_edgecolor('black')
    box.set_facecolor('white')
        # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         ax.lines[j].set_color('black')
sns.violinplot(x="TIME", y="value", data=pd.melt(df_data), color='pink')
plt.title('Distribution of station WL means for the indicated year: ec95d: ADC-OBS')
plt.ylabel('meters')
ax.set(xlabel=None)
plt.axhline(df_all.mean().mean(), c='r')
plt.gcf().subplots_adjust(bottom=0.30)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.show()
plt.savefig('yearlyMeansVsTimesViolin.png')
###############################################################################
# Distributionm of monthly mean errors for each station. 
# Add the overall total station MEAN to the plot
# The mean splot is shifted left relativer to the boxplot

# Rediculous matplotlib bug regarding shifting in the subsequrent plots - WT....
means=df_all.mean()
shiftlist = [np.nan]
[shiftlist.append(x) for x in means.to_list()]

fig, ax = plt.subplots(figsize=(12, 8))
sns.set(font_scale=1.0, style="darkgrid")
df_all.groupby(pd.Grouper(level='TIME',freq='Y')).mean().boxplot()
plt.xticks(rotation=90)
plt.title('Distribution of the mean annual WL for indicated station: ec95d. ADC-OBS') 
plt.ylabel('meters')
plt.axhline(df_all.mean().mean(), c='r')
plt.plot(shiftlist, color='r', marker='o')
#plt.show()
plt.savefig('yearlyMeans.png')

###############################################################################

# Same but X indexed by stationname: Distributionm of monthly mean errors for each station. 
# And sorted by State

# Rediculous matplotlib bug regarding shifting in the subsequrent plots - WT....
means=df_all.mean()
shiftlist = [np.nan]
[shiftlist.append(x) for x in means.to_list()]

sns.set(font_scale=1.0, style="darkgrid")
fig, ax = plt.subplots(figsize=(12, 8))
snames = [df_meta.loc[int(n)]['stationname'].replace(' ',' ') for n in df_all.columns.to_list()]
df_all_names=df_all.copy()
df_all_names.columns=snames
df_all_names.groupby(pd.Grouper(level='TIME',freq='Y')).mean().boxplot()
plt.xticks(rotation=90)
plt.title('Distribution of the mean annual WL for indicated station: ec95d. ADC-OBS')
plt.ylabel('meters')
plt.axhline(df_all.mean().mean(), c='r')
plt.gcf().subplots_adjust(bottom=0.40)
plt.plot(shiftlist, color='r', marker='o')
plt.savefig('yearlyMeansNames.png')
#plt.show()

###############################################################################
# station TOTAL means for the full time series
# No annual averaging

#fig, ax = plt.subplots(figsize=(12, 8))
#sns.set(font_scale=1.0, style="darkgrid")
#df_all.groupby(pd.Grouper(level='TIME',freq='h')).mean().boxplot()
#df_all.mean().to_frame().T.plot()
#plt.xticks(rotation=45)
#plt.title('Distribution of overall WL for indicated station: ec95d. ADC-OBS')
#plt.ylabel('meters')
#plt.axhline(df_all.mean().mean(), c='r')
#plt.show()
#plt.savefig('totalMeans.png')

###############################################################################

# How many stations for a given (partial) year

fig, ax = plt.subplots(figsize=(12, 8))
stationPerTime= df_all.groupby(pd.Grouper(level='TIME',freq='Y')).count()
stationPerYear=stationPerTime.T.astype(bool).sum(axis=0)
sns.set(font_scale=1.0, style="darkgrid")
sns.lineplot(data=stationPerYear)
#ax.legend(loc = 4,fontsize = 10)
ax.set_ylabel('Number of stations')
ax.get_xaxis().set_visible(True)
ax.set(xlabel=None)
ax.set_title('Number of NOAA-COOPS gauges: ec95d grid')
ax.grid(linestyle='-', linewidth='0.5', color='gray')
#plt.show()
plt.savefig('stationsPerYear.png')

###############################################################################
# Single station analysis
# Compare one station to its ADC and OBS plots.

yearlyERR=list()
yearlyADC=list()
yearlyOBS=list()

for year in years:
    print('Processing year {}'.format(year))
    fulldirectory=basedirectory+'/YEARLY-'+str(year)
    df = pd.read_csv('/'.join([fulldirectory,basefilename]), header=0)
    yearlyMeta.append(fetch_metadata('/'.join([fulldirectory,metafilename])))
    yearlyERR.append(fetchDataFrameSource(df, name='ERR'))
    yearlyADC.append(fetchDataFrameSource(df, name='ADC'))
    yearlyOBS.append(fetchDataFrameSource(df, name='OBS'))

df_all_ADC = pd.concat(yearlyADC)
df_all_OBS = pd.concat(yearlyOBS)
df_all_ERR = pd.concat(yearlyERR)

df_all_ERR.to_pickle('40dataAllStationsERR.pkl')
df_all_ADC.to_pickle('40dataAllStationsADC.pkl')
df_all_OBS.to_pickle('40dataAllStationsOBS.pkl')

#################################################################################
##
## Could consider iterating over all stations here
##

# Pick a station 8761724=Grand Isle 8534720=Atlantic City , 8658163=Wrightsville Beach, 8651370=Duck Pier
# NOTE: Hourly data for 8658163 is not avail from the coops.
#station=8658163
#station=8651370 # Duck pier
#station=8534720
#station=8761724

print('Process All stations')
stations = df_meta.index.to_list()
for station in stations:
    print('Processing station {}'.format(station))
    stationName=df_meta.loc[station]['stationname']
    df_ADC=df_all_ADC[str(station)].to_frame()
    df_OBS=df_all_OBS[str(station)].to_frame()
    df_ERR=df_all_ERR[str(station)].to_frame()
    ##################
    # Perform a simple line plot much like done for APSVIZ
    # noFitStationPlot(station, stationName,df_ADC, df_OBS, df_ERR)
    ##################
    fitStationPlot(station, stationName, df_ADC, df_OBS, df_ERR)

print('Finished')
