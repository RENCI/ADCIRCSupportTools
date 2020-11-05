#!/usr/bin/env python

##
## Grab the META data PKL generated from GetObsStations: (metapkl)
## Grab the Merged ADC,OBS,ERR time series data CSV computed by (mergedf) 
## Specify list up to four stations for comparison
##
import os,sys
import numpy as np
import pandas as pd
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns
from utilities.utilities import utilities

def getBounds(df):
    """
    Looks into every variables and find ther global min and global max
    I want a symmetric-about zero y axis
    """
    ymin = abs(df.min().min())
    ymax = abs(df.max().max())
    val = ymax if ymax > ymin else ymin
    ymin = -math.ceil(ymin)
    ymax = -ymin
    return ymin, ymax

def addPlot(fig, station, stationid, df_concat, variables):
    #formatter = matplotlib.dates.DateFormatter('%H-%M-%S')
    ymin, ymax = getBounds(df_concat)
    colorlist, dashlist = varToStyle(variables)
    #dashes=[(1,0),(1,0),(3,3),(2,2)]
    #print('colors {}'.format(colorlist))
    #print('dashed {}'.format(dashlist))
    nCols = len(variables)
    sns.set_style('darkgrid')
    ax=sns.lineplot(data=df_concat, palette=colorlist, dashes=dashlist)
    #ax.legend(loc ='lower left',ncol=nCols, fontsize = 8)
    ax.legend(loc = 4,fontsize = 10)
    ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
    ax.set_ylim([ymin, ymax])
    ax.get_xaxis().set_visible(True)
    ax.set(xlabel=None)
    ax.set_title('Time range {} to {}'.format( df_concat.index.min(),df_concat.index.max()),fontdict={'fontsize': 10, 'fontweight': 'medium'})
    ax.grid(linestyle='-', linewidth='0.5', color='gray')
    plt.setp(ax.get_xticklabels(), rotation = 15)
    fig.suptitle('Station Name={}'.format(station))

def convertToDF(compDict,var):
    df=pd.DataFrame(compDict[var])
    df.set_index('TIME',inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns=[''.join([var,'WL'])]
    return df

def makeDict( station, df_meta, pngfile):
    lon = df_meta.loc[int(station)]['lon']
    lat = df_meta.loc[int(station)]['lat']
    node = df_meta.loc[int(station)]['Node']
    return {'LAT':str(lat), 'LON':str(lon), 'NODE':str(node), 'FILENAME':pngfile}

## Some attempt at managing colors.
## For now do not change the order of these

colordict=dict()
colordict['forecast']='b'
colordict['nowcast']='gray'
colordict['adc']='b'
colordict['obs']='g'
colordict['err']='r'
colordict['misc']='o'

dashdict=dict()
dashdict['forecast']=(3,1)
dashdict['nowcast']=(3,1)
dashdict['adc']=(1,0)
dashdict['obs']=(1,0)
dashdict['err']=(1,0)
dashdict['misc']=(3,2)

## Now a string manipulator to help fiund the proper color
## Uses a greedy approach
def varToStyle(varlist):
    """
    Choose a color and dash style depending on variable name
    return each as ordered lists. Dash list elements are tuples
    """
    colorlist = list()
    listcolors = list(colordict.keys())
    for var in varlist:
        testvar = var.lower() 
        for test in listcolors:
            if test in testvar:
                colorlist.append(colordict[test])
                break
    listdashes = list(dashdict.keys())
    dashlist=list()
    for var in varlist:
        testvar = var.lower()
        for test in listdashes:
            if test in testvar:
                dashlist.append(dashdict[test])
                break;
    return colorlist, dashlist

## Aligned nowcast data and observations
## NOTE METADATA IS MANDATORY BUT NOT CHECKED

# NOTIDAL
computeJson='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ADDA_202010221800_202010261800_NOTIDAL/errorfield/adc_obs_error_merged_202010221800_202010261800.json'

#meta='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ADDA_202010221800_202010261800_NOTIDAL/obspkl/obs_wl_metadata_202010221800_202010261800.pkl'
#
meta='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ADDA_202010221800_202010261800_NOTIDAL/obspkl/obs_wl_metadata_202010221800_202010261800.json'

## Added ADCIRC forecast data 
adcircJson='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/Forecast/adc_forecast.json'

####
## The variable names must be unique else key creation will be a problem

## Prepend an EARLY foracast instead of afteer the compErr ran ge
#adcircJsonpre='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ForecastPrepend/adc_forecast.json'

## Prepend the NOWCAST that overlaps the prepend forecast
#adcircJsonNow='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/PreviousNowcastPrepend/adc_forecast.json'

# Buld a dict to mimic a user inputting the data

files=dict()
files['META']=meta
files['DIFFS']=computeJson
files['FORECAST']=adcircJson
#files['PREPEND']=adcircJsonpre
#files['NOWPREPEND']=adcircJsonNow

# TIMEs will not be in datetime format because of how jsons must get saved
## Start the work

listDicts = list()
for key,val in files.items():
    print(key)
    if key=='META':
        #df_meta=pd.read_pickle(val)
        #df_meta.set_index('stationid',inplace=True)
        with open(val, 'r') as fp:
            metaDict = json.load(fp) 
            df_meta=pd.DataFrame(metaDict)
            df_meta.set_index('stationid',inplace=True)
    else:
        with open(val, 'r') as fp1:
            listDicts.append(json.load(fp1))

# Double check station concordance
print('check station lists')
stations = listDicts[0]
for i in range(0,len(listDicts)):
    stations = stations & listDicts[i].keys()

print('Total intersected stations {}'.format(len(stations)))

newDict = {}
for station in stations:
    newDict[station]=listDicts[0][station]
    for dicts in range(1,len(listDicts)):
        newDict[station].update(listDicts[dicts][station])

# Now grab the list of variables (eg ADC, OBS etc). Only need to choose a single station
# How to handle DUPLICATE variable names?

variables = list(newDict[station].keys()) 
print('Set of variables is {}'.format(variables))

# Need to check if any variables are all nans. if so, causes plottiung problems. check all stations
# if np.isnan(compDicct = {}

print(' Station nan check {}'.format(station))
for station in stations:
    for variable in variables:
        if all(np.isnan(newDict[station][variable]['WL'])):
            print('Removing a station for nans {}'.format(station))
            del newDict[station]
            break

# Recheck station lisat
stations = list(newDict.keys())
print('Total number of non nan-empty  stations is {}'.format(len(stations)))

#for sntation in stations:
#station='8779748'
#station='8658163'
#TODO correct this station as either a str or an int
#stations=['8764314']

    
runProps = dict()
for station in stations:
    print('start station {}'.format(station))
    plt.close()
    stationName = df_meta.loc[int(station)]['stationname']
    listdfs = list()
    for var in variables:
        listdfs.append(convertToDF(newDict[station],var))
    df_concat = pd.concat(listdfs,axis=1)
    new_variables = df_concat.columns.to_list()
    # A per-station plot
    fig = plt.figure()
    addPlot(fig, stationName, station, df_concat, new_variables)
    pngfile='_'.join([station,'WL.png'])
    plt.savefig(pngfile)
    # Create a dict of lons,lats,nodes,filenames 
    runProps[station]=makeDict(station, df_meta, pngfile)
    #plt.show()

dictfile='test.json'
utilities.write_json_file(runProps, dictfile)
