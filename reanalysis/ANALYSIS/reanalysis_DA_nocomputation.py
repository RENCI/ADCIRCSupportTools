#!/usr/bin/env python
##
## Preliminary script to generate all station plots
##


##
## Plot interesting data based only on the pipeline generated data sets
##

import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.dates as mdates
import seaborn as sns
sns.set()

sys.path.append("/home/bblanton/utilities/")
import utilities

print("pandas Version = {}".format(pd.__version__))
print("seaborn Version = {}".format(sns.__version__))
print("numpy Version = {}".format(np.__version__))

# Stations to keep for analysis
#stationids=['8534720', '8658163', '8670870'] # ids_names=[‘Atlantic City, NJ’,‘Wrightsville Beach, NC’,‘Fort Pulaski, GA’]
#stationids=['8410140'] # random pick
#teststation=str(8658163) # Atlantic City
#teststation=str(8670870) # Fort Pulaski
#teststation=str(8658163) # Wrightsville

#################################################################################
# SHould only need to change the filenames and the NAME of the cutoff

# Error data from the prior and posterior runs
YEARLY='YEARLY-2018' # or YEARLY-2018DA
dir='/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/REANALYSIS_COMPREHENSIVE_REGION3/'+YEARLY
f=dir+'/adc_obs_error_merged.csv'
YEARLY='YEARLY-2018DA' # or YEARLY-2018DA
dirDA='/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/REANALYSIS_COMPREHENSIVE_REGION3/'+YEARLY
fDA=dirDA+'/adc_obs_error_merged.csv'
print(f)
print(fDA)

# Now select the LOWPASS filter type used to improvinmg the posteriort runs
LOWDIR=dir+'/'+'DAILY-2018YEAR-12MONTH-REGION3-RANGE8-SILL0.015-NUGGET0.005-LP48'
cutoff='48'

flowpass=LOWDIR+'/df_err_all_lowpass.pkl'
#################################################################################


# Get the station metadata for the same grid
fmeta=dir+'/obs_water_level_metadata.pkl'
df_meta=pd.read_pickle(fmeta)
df_meta.set_index('stationid',inplace=True)
print(df_meta.shape)

df = pd.read_csv(f,header=0, index_col=0)
df.index = pd.to_datetime(df.index)
dfDA = pd.read_csv(fDA,header=0, index_col=0)
dfDA.index = pd.to_datetime(dfDA.index)
dfLP = pd.read_pickle(flowpass)

# Choose range of times to keep. # Currently the DA data only goes to May
begin_date, end_date = '2018-01-01', '2018-05-01'

# Winnow the data by times
df = df.loc[begin_date:end_date]
dfDA = dfDA.loc[begin_date:end_date]
dfLP = dfLP.loc[begin_date:end_date]
#print(df)
#print(dfDA)

# Split out the individual components
df_adc_all = df[df['SRC']=='ADC'].copy()
df_obs_all = df[df['SRC']=='OBS'].copy()
df_err_all = df[df['SRC']=='ERR'].copy()
df_adc_all.drop('SRC',inplace=True,axis=1)
df_obs_all.drop('SRC',inplace=True,axis=1)
df_err_all.drop('SRC',inplace=True,axis=1)
#
dfDA_adc_all = dfDA[dfDA['SRC']=='ADC'].copy()
dfDA_obs_all = dfDA[dfDA['SRC']=='OBS'].copy()
dfDA_err_all = dfDA[dfDA['SRC']=='ERR'].copy()
dfDA_adc_all.drop('SRC',inplace=True,axis=1)
dfDA_obs_all.drop('SRC',inplace=True,axis=1)
dfDA_err_all.drop('SRC',inplace=True,axis=1)
#print(df_err_all)

# Set plotting parameters
col = sns.color_palette("bright")[0:4]
print(col)

# Remove "completely empty" stations from the data set. Don't worry if df <> daDA
df_err_all.dropna(how='all', axis=1, inplace=True)
dfDA_err_all.dropna(how='all', axis=1, inplace=True)
#print(df_err_all['8534720'])
#df_err_all['8534720'].dropna()

# Begin looping over all stations

allstations = df_err_all.columns.astype(str) # SImply to mimic what the notebook was doing
for teststation in allstations:
    stationName=df_meta.loc[int(teststation)]['stationname']
    outfile=(stationName+'timeseriesComparison.png').replace(" ", "")
    print('Processing station {} with name {}'.format(teststation, stationName))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,6), dpi=144, sharex=True)
    df_err_all[teststation].plot(ax=ax,linewidth=.5,color=col[0], label='ADCIRC - Prior')
    dfDA_err_all[teststation].plot(ax=ax,linewidth=.5,color=col[1], label='ADCIRC - Posterior')
    dfLP[teststation].plot(ax=ax,linewidth=3.0,color=col[2], label='Low pass (h) '+cutoff)
    df_obs_all[teststation].plot(ax=ax,linewidth=.5,color='gray', label='NOAA/NOS')
    ax.legend()
    ax.set_ylabel(r"$\Delta$Water Level [m]")
    ax.set_ylim(np.array([-1,1])*2)
    ax.set_title(stationName+ "  "  + teststation)
    ax.axhline(0,color='k',linewidth=.5)
    plt.savefig(outfile)
    print('Saved plot {}'.format(outfile))

# Compare distributions
for teststation in allstations:
    fig.clear()
    stationName=df_meta.loc[int(teststation)]['stationname']
    outfile=(stationName+'distributionComparison.png').replace(" ", "")
    print('Compute distributions')
    print(teststation)
    sns.distplot(df_err_all[teststation])
    sns.distplot(dfDA_err_all[teststation])
    priorm=round(df_err_all[teststation].mean(),3)
    priors=round(df_err_all[teststation].std(),3)
    postm=round(dfDA_err_all[teststation].mean(),3)
    posts=round(dfDA_err_all[teststation].std(),3)
    print('Prior error mean {}, std {}'.format(df_err_all[teststation].mean(),df_err_all[teststation].std()))
    print('Posterior error mean {}, std {}'.format(dfDA_err_all[teststation].mean(),dfDA_err_all[teststation].std()))
    anno='Prior error mean {}, std {}'.format(priorm,priors)
    annoDA='Posterior error mean {}, std {}'.format(postm,posts)
    plt.xlabel(r"$\Delta$ Water Level [m]-"+stationName)
    plt.ylabel('Frequency')
    plt.ylim(0,5)
    plt.xlim(-.8,.8)
    plt.axhline(0,color='k',linewidth=.5)
    plt.savefig(outfile)
    print('Saved distribution {}'.format(outfile))

# Skip QQ plots

# Plot DIFF of Prior-Posterior vs, input lowpass and OBS
for teststation in allstations:
    stationName=df_meta.loc[int(teststation)]['stationname']
    outfile=(stationName+'PriorDiffPosterior.png').replace(" ", "")
    # Plot the diff of raw-error prior - DA posterior
    dd=df_err_all[teststation]-dfDA_err_all[teststation]
    stationName=df_meta.loc[int(teststation)]['stationname']
    print(stationName)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,6), dpi=144, sharex=True)
    dd.plot(ax=ax,linewidth=.5,color=col[0], label=r"$\Delta$Errors: Prior - Posterior")
    dfLP[teststation].plot(ax=ax,linewidth=3.0,color=col[2], label='Low pass for kriging')
    df_obs_all[teststation].plot(ax=ax,linewidth=.5,color='gray', label='NOAA/NOS')
    ax.legend()
    ax.set_ylabel('Water Level [m]')
    ax.set_ylim(np.array([-1,1])*2)
    ax.set_title(stationName+ "  "  + teststation)
    ax.axhline(0,color='k',linewidth=.5)
    plt.savefig(outfile)
    print('Saved diffs {}'.format(outfile))

