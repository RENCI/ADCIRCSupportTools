import os,sys
import pandas as pd
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from utilities.utilities import utilities as utilities

config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
iometadata=''

rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADCIRC')
print('rootdir {}'.format(rootdir))

dtime1='2020-09-17 12'
dtime2='2020-09-21 18'

# get station-to-node indices for grid
obsyamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml')
obs_utilities = utilities.load_config(obsyamlname)
station_df = utilities.get_station_list()
station_ids = station_df["stationid"].values.reshape(-1,)
node_idx = station_df["Node"].values

adcyamlfile = os.path.join(os.path.dirname(__file__), '../../config', 'adc.yml')
adc = Adcirc(yamlname=adcyamlfile)

# Possible choices fore setting times
#adc.set_times(dtime1='2020-09-17 12', dtime2='2020-09-21 18')
#adc.set_times()
adc.set_times(dtime2='2020-09-21 18', doffset=-2)

utilities.log.info("T1 (start) = {}".format(adc.T1))
utilities.log.info("T2 (end)   = {}".format(adc.T2))

adc.get_urls()
if not any(adc.urls):
    utilities.log.abort('No URL entries. Aborting {}'.format(adc.urls))

## Not needed here
##adc.get_grid_coords()  # populates the gridx,gridy terms
##adcirc_gridx = adc.gridx[:]
##adcirc_gridy = adc.gridy[:]

#iometadata = adc.T1.strftime('%Y%m%d%H%M')+'_'+adc.T2.strftime('%Y%m%d%H%M') # Used for all classes downstream

config = utilities.load_config()
ADCdir = rootdir
ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'

if not os.path.exists(ADCfile):
    df  = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    df.to_pickle(ADCfile)
else:
    utilities.log.info('adc_wl'+iometadata+'.pkl exists.  Using that...')
    utilities.log.info(ADCfile)
    df = pd.read_pickle(ADCfile)

utilities.log.info('Done ADCIRC Reads')

