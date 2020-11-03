#!/usr/bin/env python

##
## Grab the URL(s) from the input urls.json file. Pass to ADCIRC to get the water levels
## Cast the dat ainto the APSVIZ Dict format and save off as a JSON for subsequent merging
## with other jsons to create station level plots
##

##
## This is not a special generator of ADC data. You could call GetADCIRC directly.

import os,sys
import numpy as np
import pandas as pd
import json
from utilities.utilities import utilities as utilities
from get_adcirc.GetADCIRC import Adcirc, writeToJSON, get_water_levels63

def exec_adcirc(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids):
    # Start the fetch of ADCIRC data
    adc = Adcirc(yamlname=adc_yamlname)
    adc.urls = urls
    utilities.log.info("List of available urls input specification:")
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    df.to_pickle(ADCfile)
    return ADCfile, adc.T1, adc.T2

##
## Grab the forecast and create a json file of a form for APSVIZ
##

def main(args):

    iosubdir = args.iosubdir if args.iosubdir!=None else 'JUNK'
    iometadata = args.iometadata if args.iometadata!=None else ''
    urljson = args.urljson
    adcirc_url= args.urljson
    variableName = args.variableName
    adcyamlfile=args.adcYamlname

    # Get rootdir in the usiual way
    config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
    rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='')

    # First get the set of stations and nodeids for calling ADCIRC
    utilities.log.info('Fetch YML station data')
    obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
    obs_config = utilities.load_config(obs_yamlname)
    station_df = utilities.get_station_list()
    station_id = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values

    # Second let's get the ADCIRC forecast data
    urljson = adcirc_url
    if urljson==None:
        utilities.log.error('No URL json supplied')

    if not os.path.exists(urljson):
        utilities.log.error('urljson file not found.')
        sys.exit(1)
    
    urls = utilities.read_json_file(urljson) # Can we have more than one ?
    utilities.log.info('Explicit JSON URLs provided {}'.format(urls))

    # Third get the ADCIRC data (presumably a forecast)
    utilities.log.info('Fetch ADCIRC')
    if adcyamlfile==None:
        adcyamlfile = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
    #adc_config = utilities.load_config(adcyamlfile)
    utilities.log.info('Pass YML name to ADCIRC {}'.format(adcyamlfile))
    ADCfile, timestart, timeend = exec_adcirc(urls, rootdir, iometadata, adcyamlfile, node_idx, station_id)
    utilities.log.info('Completed ADCIRC Reads')
    df_adcirc=pd.read_pickle(ADCfile)
    
    # Store as a JSON
    jsonfilename=writeToJSON(df_adcirc, rootdir, iometadata, variableName=variableName)
    utilities.log.info('Wrote ADC WL as a JSON {}'.format(jsonfilename))

    utilities.log.info('Finished')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description=main.__doc__)
    #parser.add_argument('--doffset', default=None, help='Day lag or datetime string for analysis: def to YML -4', type=int)
    #parser.add_argument('--timeout', default=None, help='YYYY-mm-dd HH:MM. Latest day of analysis def to now()', type=str)
    #parser.add_argument('--timein', default=None, help='YYYY-mm-dd HH:MM.Starting day of analysis. def to dtime2 YTML value (-4)', type=str)
    parser.add_argument('--iometadata', action='store', dest='iometadata',default=None, help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default=None, help='Used to locate output files into subdir', type=str)
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    parser.add_argument('--variableName', action='store', dest='variableName', default=None,
                        help='String: Name used in DICT/JSon to identify ADCIRC type (eg nowcast,forecast)')
    parser.add_argument('--adcYamlname', action='store', dest='adcYamlname', default=None,
                        help='String: FQFN  of alternative config to adc.yml')
    args = parser.parse_args()

    sys.exit(main(args))


