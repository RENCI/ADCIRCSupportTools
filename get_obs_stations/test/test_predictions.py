#
# Simple tests to access some of the ADCIRCDataAssimilation
# retrieveRealWaterLevels class functionality. Hard fail test cause pycharm to stop. So we collect those at the end
#

import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities

timein = '2020-01-01 00:00'
timeout = '2020-01-10 00:06'

#timein = '2018-01-01 00:00'
#timeout = '2018-12-31 18:00'

iometadata = ('_'+timein+'_'+timeout).replace(' ','_')

config = utilities.load_config() # Defaults to main.yml as sapecified in the config
rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

#rpl = GetObsStations(product='predictions', rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml'), metadata=iometadata)
rpl = GetObsStations(product='hourly_height', rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml'), metadata=iometadata)
df_stationNodelist = rpl.fetchStationNodeList()
stations = df_stationNodelist['stationid'].to_list()

df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)

#df_detailed, count_nan, stationlist, excludelist = rpl.fetchStationProductFromIDlist(timein, timeout)
df_detailed, count_nan, stationlist, excludelist = rpl.fetchStationProductFromIDlist(timein, timeout,interval='h')

#df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
#retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
#listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)

#detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = rpl.fetchOutputNames()
#utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {} detailedJ {}, smoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ ))
#print('Finished with OBS pipeline')

