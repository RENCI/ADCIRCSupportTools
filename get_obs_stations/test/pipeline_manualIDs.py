#
# Simple tests to access some of the ADCIRCDataAssimilation
# retrieveRealWaterLevels class functionality. Hard fail test cause pycharm to stop. So we collect those at the end
#

import sys,os
import numpy as np
import pandas as pd
from get_obs_stations.GetObsStations import GetObsStations
from utilities.utilities import utilities as utilities

stations = [8652587, 8654467, 8656483, 8658120, 8658163, 8775241, 8775870, 8779748]
timein = '2020-01-01 00:00'
timeout = '2020-01-10 00:06'
iometadata = ''

config = utilities.load_config() # Defaults to main.yml as sapecified in the config
rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

rpl = GetObsStations(rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../../config', 'obs.yml'), metadata=iometadata)
detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ  = rpl.executeBasicPipeline(timein, timeout)

utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv))

