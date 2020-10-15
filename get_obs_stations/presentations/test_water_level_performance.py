#
# Simple tests to access some of the ADCIRCDataAssimilation
# retrieveRealWaterLevels class functionality. Hard fail test cause pycharm to stop. So we collect those at the end
#

import time as tm
import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities

timedata = list()

for itimes in range(0,10):
    timein = '2020-01-01-00:00'
    timeout = '2020-01-10-00:06'
    iometadata = '_'+timein+'_'+timeout
    t0 = tm.time()
    config = utilities.load_config() # Defaults to main.yml as sapecified in the config
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='PerformanceObs')
    rpl = GetObsStations(rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml'), metadata=iometadata)
    stations = rpl.fetchStationNodeList()
    df_stationNodelist = rpl.fetchStationNodeList()
    stations = df_stationNodelist['stationid'].to_list()
    df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)
    df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
    listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv = rpl.fetchOutputNames()
    utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv))
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

