#!/usr/bin/env python

##
## Grab the META data PKL generated from GetObsStations: (metapkl)
## Grab the Period Error Averages data (CSV) generated from computeErrorField: (stationPeriodAves.csv)
## Grab the Station meta data CSV provided to the pipeline (CERA_NOAA_HSOFS_stations_V2.csv')
## Grab the Merged ADC,OBS,ERR time series data CSV computed by (mergedf) 
## Specify list up to four stations for comparison
##

import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fl='/projects/sequence_analysis/vol1/prediction_work/ADDA_202005011200_202005051200/errorfield/stationPeriodAves_202005011200_202005051200.csv'
extraf='/home/jtilson/ADCIRCDataAssimilation/config/CERA_NOAA_HSOFS_stations_V2.csv'
meta = '/projects/sequence_analysis/vol1/prediction_work/ADDA_202005011200_202005051200/obspkl/obs_wl_metadata_202005011200_202005051200.pkl'
adcdata='/projects/sequence_analysis/vol1/prediction_work/ADDA_202005011200_202005051200/errorfield/adc_obs_error_merged_202005011200_202005051200.csv'

df = pd.read_csv(fl,header=0, index_col=0)
dfextra=pd.read_csv(extraf,header=0, index_col=0,skiprows=[1])
df_meta=pd.read_pickle(meta)
df_meta.set_index('stationid',inplace=True)

dfextra.set_index('stationid',inplace=True)
df_states=dfextra['state']

dropList = ['lon','lat','Node','states','stationname']
df['states']=df_states
df['stationname']=dfextra['stationname']

## Grab adc data just for building titles

df_adc = pd.read_csv(adcdata,header=0, index_col=0)
df_adc = df_adc[df_adc['SRC']=='ADC']

dropList = ['lon','lat','Node','states','stationname']
df_plotData = df.drop(dropList,axis=1)

newcolumns=['[-48,-36]','[-36,-24]','[-24,-12]','[-12,now]']
df_plotData.columns=newcolumns
bplot = sns.boxplot(
                 data=df_plotData,
                 width=0.5,
                 palette="colorblind")
bplot = sns.swarmplot(
              data=df_plotData,
              color='black',
              alpha=0.25)
bplot.set_ylabel(r'$\overline{\rm  \Delta WL}$ (m) ')
bplot.set_xticklabels(bplot.get_xticklabels())
bplot.set_xlabel("Stations",fontsize=6)
bplot.set_xlabel('')
bplot.set_title('Time range {} to {}'.format( df_adc.index.min(),df_adc.index.max()))
plt.savefig('PeriodAves_aggregatedStations.png')
plt.show()

