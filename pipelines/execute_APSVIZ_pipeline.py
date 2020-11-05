#!/usr/bin/env python

###################################################################
##
## Pipeline that is useful in the APSVIZ context. Entry an adcirc URL and the nodelist
## and generate the ADS,OBS,ERR data sets.
##
###################################################################
import os,sys
import time as tm
import pandas as pd
import json

from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from get_obs_stations.GetObsStations import GetObsStations
from compute_error_field.computeErrorField import computeErrorField
from utilities.utilities import utilities as utilities


## Invoke a basic pipeline. 

def exec_adcirc(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids):
    # Start the fetch of ADCIRC data
    adc = Adcirc(adc_yamlname)
    adc.urls = urls
    utilities.log.info("List of available urls input specification:")
    ## Meaningless utilities.log.info('Observed TIMES are T1 {}, T2 {}'.format(adc.T1.strftime('%Y%m%d%H%M'), adc.T2.strftime('%Y%m%d%H%M')))

    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    if not os.path.exists(ADCfile):
        df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
        adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
        adc.T2 = df.index[-1]
        df.to_pickle(ADCfile)
        df.to_json(ADCjson)
    else:
        utilities.log.info('adc_wl'+iometadata+'.pkl exists.  Using that...')
        utilities.log.info(ADCfile)
        df = pd.read_pickle(ADCfile)
        adc.T1 = df.index[0]
        adc.T2 = df.index[-1]
        df.to_json(ADCjson)
        utilities.log.info('read times from existing pkl {}, {}'.format( adc.T1,adc.T2))
    #timestart = adc.T1.strftime('%Y%m%d%H%M')
    #timeend = adc.T2.strftime('%Y%m%d%H%M')
    timestart = adc.T1
    timeend = adc.T2
    return ADCfile, ADCjson, timestart, timeend

def exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata):
    rpl = GetObsStations(iosubdir='', rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata)
    df_stationNodelist = rpl.fetchStationNodeList()
    stations = df_stationNodelist['stationid'].to_list()
    utilities.log.info('Grabing station list from OBS YML')
    df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stations)
    df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
    retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
    listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = rpl.fetchOutputNames()
    return detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ 

def exec_error(obsf, adcf, meta, err_yamlname, rootdir, iometadata): 
    cmp = computeErrorField(obsf, adcf, meta, yamlname=err_yamlname, rootdir=rootdir)
    cmp.executePipelineNoTidalTransform(metadata=iometadata,subdir='')
    errf, finalf, cyclef, metaf, mergedf,jsonf = cmp._fetchOutputFilenames()
    return errf, finalf, cyclef, metaf, mergedf, jsonf

# noinspection PyPep8Naming,DuplicatedCode
def main(args):

    t0 = tm.time()
    outfiles = dict()

    # Get input adcirc url and check for existance
    urljson = args.urljson
    if not os.path.exists(urljson):
        utilities.log.error('urljson file not found.')
        sys.exit(1)
    urls = utilities.read_json_file(urljson) # Can we have more than one ?
    utilities.log.info('Explicit JSON URLs provided {}'.format(urls))

    # 1) Setup main config data
    iosubdir = args.iosubdir
    iometadata = args.iometadata
    main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
    rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
    utilities.log.info('Specified rootdir underwhich all files wil; be stored. Rootdir is {}'.format(rootdir))

    # 2) Setup ADCIRC specific YML-resident inputs
    # Such as node_idx data
    utilities.log.info('Fetch OBS station data')
    obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
    obs_config = utilities.load_config(obs_yamlname)
    station_df = utilities.get_station_list()
    station_ids = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values

    utilities.log.info('Fetch ADCIRC')
    adc_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
    adc_config = utilities.load_config(adc_yamlname)
    ADCfile, ADCjson, timestart, timeend = exec_adcirc(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids)
    utilities.log.info('Completed ADCIRC Reads')
    outfiles['ADCIRC_WL_PKL']=ADCfile
    outfiles['ADCIRC_WL_JSON']=ADCjson

    # 3) Setup OBS specific YML-resident values
    utilities.log.info('Fetch Observations')
    #obs_yamlname = os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'obs.yml')

    # Grab time Range and tentative station list from the ADCIRC fetch  (stations may still be filtered out)
    timein = timestart.strftime('%Y%m%d %H:%M')
    timeout = timeend.strftime('%Y%m%d %H:%M')
    utilities.log.info('ADC provided times are {} and {}'.format(timein, timeout))

    # Could also set stations to None
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata)
    outfiles['OBS_DETAILED_PKL']=detailedpkl
    outfiles['OBS_SMOOTHED_PKL']=smoothedpkl
    outfiles['OBS_METADATA_PKL']=metapkl
    outfiles['OBS_NOAA_COOPS_URLS_CSV']=urlcsv
    outfiles['OBS_EXCLUDED_CSV']=exccsv
    outfiles['OBS_DETAILED_JSON']=detailedJ
    outfiles['OBS_SMOOTHED_JSON']=smoothedJ
    outfiles['OBS_METADATA_JSON']=metaJ
    utilities.log.info('Completed OBS: Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {}, DetailedJ {}, SmoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ))

    # 4) Setup ERR specific YML-resident values
    utilities.log.info('Error computation')
    err_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'err.yml')
    meta = outfiles['OBS_METADATA_PKL']
    obsf = outfiles['OBS_SMOOTHED_PKL']
    adcf = outfiles['ADCIRC_WL_PKL']

    errf, finalf, cyclef, metaf, mergedf, jsonf = exec_error(obsf, adcf, meta, err_yamlname, rootdir, iometadata)
    
    outfiles['ERR_TIME_PKL']=errf
    outfiles['ERR_STATION_AVES_CSV']=errf  # THis would pass to interpolator
    outfiles['ERR_STATION_PERIOD_AVES_CSV']=cyclef
    outfiles['ERR_METADATA_CSV']=metaf
    outfiles['ERR_ADCOBSERR_MERGED_CSV']=mergedf # This is useful for visualization insets of station bahavior
    utilities.log.info('Completed ERR')

    utilities.log.info('Finished pipeline in {} s'.format(tm.time()-t0))

    # Setup for computing the station diffs (aka adcirc - obs errors)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--ignore_pkl', help="Ignore existing pickle files.", action='store_true')
    parser.add_argument('--doffset', default=None, help='Day lag or datetime string for analysis: def to YML -4', type=int)
    parser.add_argument('--iometadata', action='store', dest='iometadata',default='', help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default='', help='Used to locate output files into subdir', type=str)
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    args = parser.parse_args()
    sys.exit(main(args))

