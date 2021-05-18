#!/usr/bin/env python

###################################################################
##
## Pipeline that is useful in the APSVIZ context. Entry an adcirc URL and the nodelist
## and generate the ADS,OBS,ERR data sets.
## No interpolation is performed at this stage
##
## This simply take one or more URLs as input and computes the error fields.
## For the Reanalysis work (one year in length), we "trick" the code into 
## computing errors across the entire year by modifying the err.yml params
##
###################################################################

import os,sys
import time as tm
import pandas as pd
import json
import re

from get_adcirc.GetADCIRC import Adcirc, writeToJSON, get_water_levels63
from get_obs_stations.GetObsStations import GetObsStations
from compute_error_field.computeErrorField import computeErrorField
from utilities.utilities import utilities as utilities
from visualization.stationPlotter import stationPlotter
import datetime as dt

def buildLocalConfig(n_aveper=4, n_period=24, n_tide=12.42, n_pad=1):
    """
    Temporary function to build a config object that can be passed to computeError method
    We need to beable to update AvePer and n_periods. We use these to trick the code into
    processing (eg) an entire years worth of data in 24 hour blocks.
    The defaults are the same as those used for a traditional ADDA
    """
    cfg = {'TIME': 
        {'AvgPer': n_aveper,
         'n_tide': n_tide,
         'n_period': n_period,
         'n_pad': n_pad }, 
        'ERRORFIELD': {'EX_OUTLIER': True}
    }
    utilities.log.info('Constructed an internal err.yml configuration dict containing {}'.format(cfg))
    return cfg

## Invoke a basic pipeline. 

def extractDateFromURL(url):
    """
    Dig thru url, fetch date, converet to datetime object and return
    """
    t = re.findall(r'\d{4}\d{1,2}\d{1,2}\d{1,2}', url)
    tdt = dt.datetime.strptime(t[0], '%Y%m%d%H')
    return tdt

def fetchNOW():
    """
    If you simply want the nowtime here it is.
    """
    tdt = dt.datetime.now()
    return tdt

def exec_adcirc_url(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids, grid):
    adc = Adcirc(adc_yamlname, grid=grid)
    #rootdir = utilities.fetchBasedir(inrootdir) # Ensure the directory exists
    adc.urls = urls
    adc.get_grid_coords()
    utilities.log.debug(' gridx {}'.format(adc.gridx[:]))
    utilities.log.debug(' gridy {}'.format(adc.gridy[:]))
    adcx = adc.gridx[:].tolist()
    adcy = adc.gridy[:].tolist()
    adc_coords = {'lon':adcx, 'lat':adcy}
    utilities.log.info("List of available urls input specification:")
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    ADCfilecords = rootdir+'/adc_coord'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl',iometadata=iometadata)
    print('write new json')
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl_forecast')
    utilities.write_json_file(adc_coords, ADCfilecords)
    timestart = adc.T1
    timeend = adc.T2
    utilities.log.info('Times returned by get_adcirc {},{}'.format(adc.T1,adc.T2))
    return ADCfile, ADCjson, timestart, timeend

def exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir, stationFile,knockout=None):
    rpl = GetObsStations(iosubdir=iosubdir, rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata, stationFile=stationFile, knockout=knockout)
    df_stationNodelist = rpl.fetchStationNodeList()
    stations = df_stationNodelist['stationid'].to_list()
    #utilities.log.info('Choose a limited number of stations')
    #stations = rpl.stationListFromYaml()
    #
    utilities.log.info('Grabing station list from OBS YML')
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
    #listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)
    #detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = rpl.fetchOutputNames()
    return detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ 

def exec_error(obsf, adcf, meta, local_config, rootdir, iometadata, iosubdir): 
    """
    Build a new config that rangers over the entire time range
    """
    #cmp = computeErrorField(obsf, adcf, meta, yamlname=err_yamlname, rootdir=rootdir)
    cmp = computeErrorField(obsf, adcf, meta, inputcfg=local_config, rootdir=rootdir)
    cmp.executePipelineNoTidalTransform(metadata=iometadata,subdir=iosubdir)
    errf, finalf, cyclef, metaf, mergedf,jsonf = cmp._fetchOutputFilenames()
    return errf, finalf, cyclef, metaf, mergedf, jsonf

def exec_pngs(files, rootdir, iometadata, iosubdir):
    utilities.log.info('Begin Generation of station-specific PNG insets')
    viz = stationPlotter(files=files, iosubdir=iosubdir, rootdir=rootdir, metadata=iometadata)
    png_dict = viz.generatePNGs()
    return png_dict

    
# noinspection PyPep8Naming,DuplicatedCode
def main(args):
    t0 = tm.time()
    outfiles = dict()

    doffset = args.doffset
    chosengrid=args.grid

    knockout=args.knockout

    # Get input adcirc url and check for existance
    if args.urljson != None:
        urljson = args.urljson
        if not os.path.exists(urljson):
            utilities.log.error('urljson file not found.')
            sys.exit(1)
        urls = utilities.read_json_file(urljson) # Can we have more than one ?
        if len(urls) !=1:
            utilities.log.error('JSON file can only contain a single URL. It has {}'.format(len(urls)))
        utilities.log.info('Explicit JSON URLs provided {}'.format(urls))
    elif args.url != None:
        # If here we still need to build a dict for ADCIRC
        url = args.url
        #dte = extractDateFromURL(url)
        dte='dummy'
        urls={dte:url}
        utilities.log.info('Explicit URL provided {}'.format(urls))
    else:
        utilities.log.error('No Proper URL specified')

    if knockout is not None:
        utilities.log.info('A station and time knockout file was specified {}'.format(knockout))
        dict_knockout = utilities.read_json_file(knockout)
        utilities.log.debug('Knockout dict {}'.format(dict_knockout))

    # 1) Setup main config data
    iosubdir = args.iosubdir
    iometadata = args.iometadata
    main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc

#rootdir = utilities.fetchBasedir(inrootdir) # Ensure the directory exists

    # Trick the systen to create the dirs
    if args.rootdir is None:
        inrootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'],basedirExtra='')
    else:
        inrootdir = utilities.setBasedir(args.rootdir)

    utilities.log.info('Specified rootdir underwhich all files will be stored. Rootdir is {}'.format(rootdir))

    outfiles['RUNDATE']=dt.datetime.now().strftime('%Y%m%d%H%M')
    outfiles['ROOTDIR']=rootdir
    outfiles['IOSUBDIR']=iosubdir
    outfiles['IOMETADATA']=iometadata
   
    # Such as node_idx data

    utilities.log.info('Fetch station data for grid {}'.format(chosengrid))
    try:
        stationFile=main_config['STATIONS'][chosengrid.upper()]
        stationFile=os.path.join(os.path.dirname(__file__), "../config", stationFile)
    except KeyError as e:
        utilities.log.error('ADDA: Error specifying grid. Uppercase version Not found in the main.yml {}'.format(chosengrid))
        utilities.log.error(e)
        sys.exit()

    # Read the stationids and nodeid data
    df = pd.read_csv(stationFile, index_col=0, header=0, skiprows=[1], sep=',')
    station_ids = df["stationid"].values.reshape(-1,)
    node_idx = df["Node"].values.reshape(-1,1) # had to slightly change the shaping here
    utilities.log.info('Retrived stations and nodeids from {}'.format(stationFile))

    # NOTE using inrootdir here
    utilities.log.info('Fetch ADCIRC')
    adc_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
    ADCfile, ADCjson, timestart, timeend = exec_adcirc_url(urls, inrootdir, iometadata, adc_yamlname, node_idx, station_ids, chosengrid)
    utilities.log.info('Completed ADCIRC nowcast Reads')
    outfiles['ADCIRC_WL_PKL']=ADCfile
    outfiles['ADCIRC_WL_JSON']=ADCjson

    # 3) Setup OBS specific YML-resident values
    utilities.log.info('Fetch Observations')
    obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.hsofs.yml')

    # Grab time Range and tentative station list from the ADCIRC fetch  (stations may still be filtered out)
    timein = timestart.strftime('%Y%m%d %H:%M')
    timeout = timeend.strftime('%Y%m%d %H:%M')
    utilities.log.info('ADC provided times are {} and {}'.format(timein, timeout))

    # Could also set stations to None
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir, stationFile)
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
    # utilities.log.info('Error computation NOTIDAL corrections')
    err_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'err.yml')
    meta = outfiles['OBS_METADATA_PKL']
    obsf = outfiles['OBS_SMOOTHED_PKL']
    adcf = outfiles['ADCIRC_WL_PKL']
    # Compute how many 24-hours ranges from time2 we should look back
    # A bit of a waste need to read files again
    df_temp_adc = pd.read_pickle(adcf)
    df_temp_obs = pd.read_pickle(obsf)
    n_length = len(df_temp_adc.index & df_temp_obs.index)
    n_aveper = n_length // 24
    local_config = buildLocalConfig(n_aveper=n_aveper, n_period=24)
    utilities.log.info('Config to error {}'.format(local_config))
    errf, finalf, cyclef, metaf, mergedf, jsonf = exec_error(obsf, adcf, meta, local_config, rootdir, iometadata, iosubdir)
    outfiles['ERR_TIME_PKL']=errf
    outfiles['ERR_TIME_JSON']=jsonf
    outfiles['ERR_STATION_AVES_CSV']=errf  # THis would pass to interpolator
    outfiles['ERR_STATION_PERIOD_AVES_CSV']=cyclef
    outfiles['ERR_METADATA_CSV']=metaf
    outfiles['ERR_ADCOBSERR_MERGED_CSV']=mergedf # This is useful for visualization insets of station bahavior
    utilities.log.info('Completed ERR')

    # 6) Build a series of station-PNGs.
    # Build input dict for the plotting
    files=dict()
    files['META']=outfiles['OBS_METADATA_JSON']
    files['DIFFS']=outfiles['ERR_TIME_JSON']
    utilities.log.info('PNG plotter dict is {}'.format(files))
    png_dict = exec_pngs(files=files, rootdir=rootdir, iometadata=iometadata, iosubdir=iosubdir)

    # Merge dict from plotter and finish up
    outfiles.update(png_dict)
    outfilesjson = utilities.writeDictToJson(outfiles, rootdir=rootdir,subdir=iosubdir,fileroot='runProps',iometadata='') # Never change fname
    utilities.log.info('Wrote pipeline Dict data to {}'.format(outfilesjson)) 
    utilities.log.info('Finished pipeline in {} s'.format(tm.time()-t0))
    return outfiles

    # Setup for computing the station diffs (aka adcirc - obs errors)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--rootdir', action='store', dest='rootdir', default=None,
                        help='Available high leverl directory')
    parser.add_argument('--ignore_pkl', help="Ignore existing pickle files.", action='store_true')
    parser.add_argument('--doffset', default=-4, help='Day lag or datetime string for analysis: def to YML -4', type=int)
    parser.add_argument('--iometadata', action='store', dest='iometadata',default='', help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default='', help='Used to locate output files into subdir', type=str)
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    parser.add_argument('--url', action='store', dest='url', default=None,
                        help='String: url.')
    parser.add_argument('--grid', default='hsofs',dest='grid', help='Choose name of available grid',type=str)
    parser.add_argument('--knockout', default=None, dest='knockout', help='knockout jsonfilename', type=str)
    args = parser.parse_args()
    sys.exit(main(args))

