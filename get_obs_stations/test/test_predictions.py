#
# Simple tests to access some of the ADCIRCDataAssimilation
# retrieveRealWaterLevels class functionality. Hard fail test cause pycharm to stop. So we collect those at the end
#

import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities

#timein = '2020-10-01 18:00'
#timeout = '2020-10-06 18:00'

timein = '2018-01-01 00:00'
timeout = '2018-12-31 18:00'

iometadata = ('_'+timein+'_'+timeout).replace(' ','_')

config = utilities.load_config() # Defaults to main.yml as sapecified in the config
rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

rpl = GetObsStations(product='water_level', rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml'), metadata=iometadata)
df_stationNodelist = rpl.fetchStationNodeList()
stations = df_stationNodelist['stationid'].to_list()

df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)
#df_detailed, count_nan, stationlist, excludelist = rpl.fetchStationProductFromIDlist(timein, timeout, interval='h')
df_pruned, count_nan, newAllstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)

retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
dummy = rpl.buildURLsForStationPlotting(newAllstationlist, timein, timeout) # 

#detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = rpl.fetchOutputNames()
#utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {} detailedJ {}, smoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ ))

# THis is better sionce now the write are encapsulated
outputdict = rpl.writeFilesToDisk()
print('output files {}'.format(outputdict))

detailedpkl=outputdict['PKLdetailed']
detailedJ=outputdict['JSONdetailed']
smoothedpkl=outputdict['PKLsmoothed']
smoothedJ=outputdict['JSONsmoothed']
metapkl=outputdict['PKLmeta']
metaJ=outputdict['JSONmeta']
urlcsv=outputdict['CSVurl']
exccsv=outputdict['CSVexclude']

utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {} detailedJ {}, smoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ ))

print('Finished with OBS pipeline')

