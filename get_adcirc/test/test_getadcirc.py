import os,sys
import pandas as pd
import json
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from utilities.utilities import utilities as utilities

config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
iometadata=''

rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADCIRC')
print('rootdir {}'.format(rootdir))

# get station-to-node indices for grid
obsyamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml')
obs_utilities = utilities.load_config(obsyamlname)
station_df = utilities.get_station_list()
station_ids = station_df["stationid"].values.reshape(-1,)
node_idx = station_df["Node"].values

adcyamlfile = os.path.join(os.path.dirname(__file__), '../../config', 'adc.yml')
adc = Adcirc(yamlname=adcyamlfile, metadata='junk', dtime1='2020-09-17 12', dtime2='2020-09-21 18')

# adc.set_times(dtime1='2020-09-17 12', dtime2='2020-09-21 18') # would also work 

utilities.log.info("T1 (start) = {}".format(adc.T1))
utilities.log.info("T2 (end)   = {}".format(adc.T2))

adc.get_urls()
if not any(adc.urls):
    utilities.log.abort('No URL entries. Aborting {}'.format(adc.urls))

ADCdir = rootdir
ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'
ADCfilejson = ADCdir+'/adc_wl'+iometadata+'.json'


if not os.path.exists(ADCfile):
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    df.to_pickle(ADCfile)
    df.to_json(ADCfilejson)
else:
    utilities.log.info('adc_wl'+iometadata+'.pkl exists.  Using that...')
    utilities.log.info(ADCfile)
    df = pd.read_pickle(ADCfile)

utilities.log.info('Done ADCIRC Reads')

