#
# Simple tests to access some of the ADCIRCDataAssimilation
# retrieveRealWaterLevels class functionality. Hard fail test cause pycharm to stop. So we collect those at the end
#

import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities

config = utilities.load_config() # Defaults to main.yml as sapecified in the config

chosengrid='region3'
try:
    stationFile=config['STATIONS'][chosengrid.upper()]
    stationFile=os.path.join(os.path.dirname(__file__), "../config", stationFile)
except KeyError as e:
    utilities.log.error('ADDA: Error specifying grid. Uppercase version Not found in the main.yml {}'.format(chosengrid))
    utilities.log.error(e)
    sys.exit()

utilities.log.info('stationFile is {}'.format(stationFile))
stations = [8413320, 8418150, 8652587, 8654467, 8656483, 8658120, 8658163, 8775241, 8775870, 8779748]
timein = '2018-01-01 00:00'
timeout = '2018-01-07 00:06'

iometadata = ('_'+timein+'_'+timeout).replace(' ','_')

rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

rpl = GetObsStations(iosubdir='',rootdir=rootdir, stationFile=stationFile, yamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml'), metadata=iometadata)
stations = stations + [999999]

df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)

df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
dummy = rpl.buildURLsForStationPlotting(newstationlist, timein, timeout) # Could also use newstationlist+excludelist

outputdict = rpl.writeFilesToDisk()

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

