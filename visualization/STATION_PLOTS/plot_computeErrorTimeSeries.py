#!/usr/bin/env python

##
## Grab the META data PKL generated from GetObsStations: (metapkl)
## Grab the Merged ADC,OBS,ERR time series data CSV computed by (mergedf) 
## Specify list up to four stations for comparison
##
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Plot a 2x2 grid
def addSubplot(fig, station, df_meta, df_adc_all, df_obs_all, df_err_all, index):
    df_adc = df_adc_all.loc[:,str(station)]
    df_obs = df_obs_all.loc[:,str(station)]
    df_err = df_err_all.loc[:,str(station)]
    stationName = df_meta.loc[station]['stationname']
    ax = fig.add_subplot(2, 2, index)
    df_adc.plot(color='r', label="ADC", linestyle="-")
    df_obs.plot(color='b', label="OBS", linestyle="-")
    df_err.plot(color='g', label="ERR", linestyle="-")
    ax.legend(loc ='lower left',ncol=3, fontsize = 6)
    ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel('')
    ax.get_xaxis().set_visible(False)
    ax.set_title(stationName, fontdict={'fontsize': 8, 'fontweight': 'medium'})

## Test data
##fl='/projects/sequence_analysis/vol1/prediction_work/ADDA_202005011200_202005051200/errorfield/adc_obs_error_merged_202005011200_202005051200.csv'
##meta = '/projects/sequence_analysis/vol1/prediction_work/ADDA_202005011200_202005051200/obspkl/obs_wl_metadata_202005011200_202005051200.pkl'

fl='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ADDA_202010221800_202010261800/errorfield/adc_obs_error_merged_202010221800_202010261800.csv'
meta='/projects/sequence_analysis/vol1/prediction_work/ADDA_Prediction_merges/ADDA_202010221800_202010261800/obspkl/obs_wl_metadata_202010221800_202010261800.pkl'

## Start the work

df = pd.read_csv(fl,header=0, index_col=0)
df_meta=pd.read_pickle(meta)
df_meta.set_index('stationid',inplace=True)

df_adc_all = df[df['SRC']=='ADC']
df_obs_all = df[df['SRC']=='OBS']
df_err_all = df[df['SRC']=='ERR']

stations = [8651370,8656483,8658163,8779748]
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for index,station in zip(range(1,5),stations):
    addSubplot(fig, station, df_meta, df_adc_all, df_obs_all, df_err_all, index)

fig.suptitle('Time range {} to {}'.format( df_adc_all.index.min(),df_adc_all.index.max()))
fig.set_size_inches(8, 8)
#plt.savefig('ADSOBSERR_fourStations.png')
plt.show()

