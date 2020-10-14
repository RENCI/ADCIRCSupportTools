import time as tm 
import numpy as np
import os,sys
import pandas as pd
import json
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from utilities.utilities import utilities as utilities

def str2datetime(s):
    from datetime import datetime
    try:
        d = datetime.strptime(s, "%Y%m%d-%H:%M:%S")
    except:
        utilities.log.error("Override time2 has bad format:".format(s))
    utilities.log.info("Overriding NOW as time2 specification: {}".format(d))
    return d

timedata = list()

timein = '2020-09-17 13:00'
timeout = '2020-09-21 18:00'

for itimes in range(0,10):
    t0 = tm.time()
    config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
    iometadata=''
    rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADCIRC'+str(itimes))
    adc = Adcirc()
    station_df = utilities.get_station_list()
    station_id = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values
    adc.set_times(timein, timeout)
    #adc.T1=str2datetime(timein)
    #adc.T2=str2datetime(timeout)
    utilities.log.info("T1 (start) = {}".format(adc.T1))
    utilities.log.info("T2 (end)   = {}".format(adc.T2))
    adc.get_urls()
    config = utilities.load_config()
    ADCdir = rootdir
    ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = ADCdir+'/adc_wl'+iometadata+'.json'
    if not os.path.exists(ADCfile):
        df = get_water_levels63(adc.urls, node_idx, station_id) # Gets ADCIRC water levels
        df.to_pickle(ADCfile)
        df.to_json(ADCjson)
    else:
        utilities.log.info("adc_wl.pkl exists.  Using that...")
        utilities.log.info(ADCfile)
        df = pd.read_pickle(ADCfile)
    utilities.log.info('Done ADCIRC Reads')
    timedata.append( tm.time()-t0 )

print('Finished with loop. Compute statistics.')

import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

m, mmh,mph = mean_confidence_interval(timedata)

print('Total times: Mean {} 95% CI range {}-{}'.format(m, mmh, mph))

##--timein '2020-01-01 13:00' --timeout '2020-01-10 00:06'

