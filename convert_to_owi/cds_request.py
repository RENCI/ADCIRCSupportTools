#!/usr/bin/env python

##
## Cannot call for a years worth at one time.
## So here we simply chop it up into single month retrieves.
## Based on the API call constructed by copernicus when requesting a year of data
## They specify the months but then only specify 31 days. So presumably, this does not
## cause a failure for months with < 31 days
## https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview
## Also assume we need to reconnect each month

import os,sys
import pandas as pd
import time as tm
import cdsapi
from argparse import ArgumentParser
from utilities.utilities import utilities
from requests.exceptions import ConnectionError,Timeout,HTTPError

###############################################################################
# Need this if going for the data < 1979.
#         c.retrieve(
#            'reanalysis-era5-single-levels-preliminary-back-extension', config, filename)
def wrap_retrieve(config, filename):
    month = config['month']
    print('Start month: {} outfilename is {}'.format(month, filename))
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels', config, filename)
    except ConnectionError:
        utilities.log.error('Hard fail: Could not connect to CDS for ERA5 products')
    except HTTPError:
        utilities.log.error('Hard fail: HTTP error to CDS for ERA5 products')
    except Timeout:
        utilities.log.error('Hard fail: Timeout error to CDS for ERA5 products')

## end defs
###############################################################################

# If months is a list of more than 1 elem split into monthlies.
#monthList = ['01','02','03','04','05','06','07','08','09','10','11','12']

#basedirExtra='ERA5' # Will build results into a rootdir = $RUNTIMEDSIR/ERA5

#subdir=year will push a year of monthlies into $RUNTIMEDSIR/ERA5/year

def main(args):
    #subdir='ERA5' # Leave it blank

    testyear = args.testyear
    if testyear != None:
        utilities.log.info('Overriding year value to {}'.format(testyear))
    
    month = args.month
    if month != None:
        utilities.log.info('Overriding month value to {}'.format(month))
        monthlist = [ month ] # needs to be a 2 digit string
    else:
        monthlist = cds_config['CDS']['month'] # Grab from YML else process a single month from the CLI
        utilities.log.info('Fetching month from YAML value to {}'.format(monthlist))

    basedirExtra = args.metadata # Eg ERA5/global

    utilities.log.info("CDS retriever in {}.".format(os.getcwd()))
    
    # Get basic runtime info
    config = utilities.load_config() # Defaults to main.yml as specified in the config
    # Get fetch characteristics
    cds_yaml = os.path.join(os.path.dirname(__file__), "../config/", "cds.yml")
    cds_config = utilities.load_config(cds_yaml)

    print('Before any overrides: Config {}'.format(cds_config))

    # We assume a single years worth of data in a subdirectory of monthlies
    year = cds_config['CDS']['year'] if testyear == None else testyear
    
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra=basedirExtra)

# If you forget to supply a year, CDS will report the following error:
# Ambiguous parameter: day could be DATE or DATABASE - No request

    subdir=year

    if year==None:
        utilities.log.exit('Missing year. Please either add a year keyword:value to the YAML or provide it on the CLI')

    utilities.log.debug('Input YAML file \n{}'.format(config))

    t0 = tm.time()
    for month in monthlist:
        supp_year = testyear
        utilities.log.info('Processing month {}'.format(month))
        single_config = cds_config['CDS']
        single_config['month']=month
        single_config['year']=year # Override if desired
        print('single config {}'.format(single_config))
        #outfilename =  utilities.getSubdirectoryFileName(rootdir, subdir, '_'.join([month,'download_wind.nc']))
        outfilename =  utilities.getSubdirectoryFileName(rootdir, subdir, '.'.join([month,'nc']))
        wrap_retrieve(single_config, outfilename) 
        utilities.log.info('Wrote file {} to disk'.format(outfilename))
    
    utilities.log.info('Total CDS fetch and write time is {}'.format(tm.time()-t0))
    print('Finished')

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('--year', type=str, action='store', dest='testyear', default=None,
                        help='Optionally override year value specified in the YAML file.')
    parser.add_argument('--month', type=str, action='store', dest='month', default=None,
                        help='Optionally override month values specified in the YAML file.')
    parser.add_argument('--metadata', type=str, action='store', dest='metadata', default='ERA5',
                        help='Optionally Adds extra info to the destination path $RUNTIMEDIR/metadata')
    args = parser.parse_args()
    sys.exit(main(args))
