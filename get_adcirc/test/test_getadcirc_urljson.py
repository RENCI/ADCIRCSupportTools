import os,sys
import pandas as pd
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from utilities.utilities import utilities as utilities
import json

urljson='/home/jtilson/ADCIRCSupportTools/get_adcirc/test/data.json'

config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
iometadata=''
rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADCIRC')

# get station-to-node indices for grid
obsyamlname=os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'obs.yml')
obs_utilities = utilities.load_config(obsyamlname)
station_df = utilities.get_station_list()
station_ids = station_df["stationid"].values.reshape(-1,)
node_idx = station_df["Node"].values

adcyamlfile = os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'adc.yml')

if not os.path.exists(urljson):
    utilities.log.error('urljson file not found.')
    sys.exit(1)
urls = utilities.read_json_file(urljson)
utilities.log.info('Explicit JSON URLs provided {}'.format(urls))

adc = Adcirc(yamlname=adcyamlfile)
adc.urls = urls
utilities.log.info("List of available urls input specification:")

##adc.get_grid_coords()  # populates the gridx,gridy terms
##adcirc_gridx = adc.gridx[:]
##adcirc_gridy = adc.gridy[:]
#iometadata = adc.T1.strftime('%Y%m%d%H%M')+'_'+adc.T2.strftime('%Y%m%d%H%M') # Used for all classes downstream


config = utilities.load_config()
ADCdir = rootdir
ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'
ADCjson = ADCdir+'/adc_wl'+iometadata+'.json'

if not os.path.exists(ADCfile):
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    df.to_pickle(ADCfile)
    df.to_json(ADCjson)
    adc.T1 = df.index[0]
    adc.T2 = df.index[-1]
else:
    utilities.log.info('adc_wl'+iometadata+'.pkl exists.  Using that...')
    utilities.log.info(ADCfile)
    df = pd.read_pickle(ADCfile)
    adc.T1 = df.index[0]
    adc.T2 = df.index[-1]

utilities.log.info('Done ADCIRC Reads')

