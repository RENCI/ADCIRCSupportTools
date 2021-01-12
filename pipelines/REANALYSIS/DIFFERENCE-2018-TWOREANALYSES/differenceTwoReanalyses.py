#!/usr/bin/env python

###################################################################
##
## Compare the CFS from the earlier Reanlysis to see what their diffs are.
## This version converted to new OBSrefactor API
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

def exec_adcirc(dtime2, rootdir, iometadata, adc_yamlname, node_idx, station_ids, doffset=-4):
    """
    dtime2 arrives as a string in the format YYYY-mm-dd MM:SS 
    """
    # Start the fetch of ADCIRC data
    adc = Adcirc(adc_yamlname)
    adc.set_times(dtime2=dtime2, doffset=doffset)
    utilities.log.info("T1 (start) = {}".format(adc.T1))
    utilities.log.info("T2 (end)   = {}".format(adc.T2))
    adc.get_urls()
    if not any(adc.urls):
        utilities.log.error('No URL entries. Aborting {}'.format(adc.urls))
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl',iometadata=iometadata)
    ##df.to_pickle(ADCfile)
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl')
    ##df.to_json(ADCjson)
    utilities.log.info(ADCfile)
    utilities.log.info('times {}, {}'.format( adc.T1,adc.T2))
    timestart = adc.T1
    timeend = adc.T2
    return ADCfile, ADCjson, timestart, timeend

def exec_adcirc_url(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids):
    adc = Adcirc(adc_yamlname)
    adc.urls = urls
    utilities.log.info("List of available urls input specification:")
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl_forecast',iometadata=iometadata)
    print('write new json')
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl_forecast')
    timestart = adc.T1
    timeend = adc.T2
    return ADCfile, ADCjson, timestart, timeend

def exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir):
    rpl = GetObsStations(iosubdir=iosubdir, rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata)
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
    return detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ 

def exec_error(obsf, adcf, meta, err_yamlname, rootdir, iometadata, iosubdir): 
    cmp = computeErrorField(obsf, adcf, meta, yamlname=err_yamlname, rootdir=rootdir)
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

    url_reanalysis='http://tds.renci.org:8080/thredds//dodsC/Reanalysis/ADCIRC/hsofs/fort.63.nc'
    url_cfsv2='http://tds.renci.org:8080/thredds/dodsC/Reanalysis/ADCIRC/CFSv2/hsofs/temp/fort.63.nc'

    dte='reanalysis'
    urls_reanalysis={dte:url_reanalysis}
    utilities.log.info('Reanalysis URL provided {}'.format(urls_reanalysis))

    dte='cfsv2'
    urls_cfsv2={dte:url_cfsv2}
    utilities.log.info('CFSv2 URL provided {}'.format(urls_reanalysis))


    # 1) Setup main config data
    iosubdir = args.iosubdir
    iometadata = args.iometadata
    main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc

    if args.rootdir is None:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
    else:
        rootdir = args.rootdir
    utilities.log.info('Specified rootdir underwhich all files wil; be stored. Rootdir is {}'.format(rootdir))

    outfiles['RUNDATE']=dt.datetime.now().strftime('%Y%m%d%H%M')
    outfiles['ROOTDIR']=rootdir
    outfiles['IOSUBDIR']=iosubdir
    outfiles['IOMETADATA']=iometadata
   
    # 2) Setup list of relevant stations and ADCIRC nodeids
    # Such as node_idx data
    utilities.log.info('Fetch OBS station data')
    obs_yamlname='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/config/obs.yml'
    obs_config = utilities.load_config(obs_yamlname)
    station_df = utilities.get_station_list()
    station_ids = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values

    # 3) Fetch REANALYSIS data

    utilities.log.info('Fetch ADCIRC')
    adc_yamlname='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/config/adc.yml'
    ADCfile_reanalysis, ADCjson_reanalysis, timestart_reanalysis, timeend_reanalysis = exec_adcirc_url(urls_reanalysis, rootdir, iometadata, adc_yamlname, node_idx, station_ids)
    utilities.log.info('Completed REANALYSIS ADCIRC nowcast Reads')
    outfiles['ADCIRC_REANALYSIS_WL_PKL']=ADCfile_reanalysis
    outfiles['ADCIRC_REANALYSIS_WL_JSON']=ADCjson_reanalysis

    # 4) Fetch CFSv2 data

    ADCfile_cfsv2, ADCjson_cfsv2, timestart_cfsv2, timeend_cfsv2 = exec_adcirc_url(urls_cfsv2, rootdir, iometadata, adc_yamlname, node_idx, station_ids)
    utilities.log.info('Completed CFSV2 ADCIRC nowcast Reads')
    outfiles['ADCIRC_CFSV2_WL_PKL']=ADCfile_cfsv2
    outfiles['ADCIRC_CFSV2_WL_JSON']=ADCjson_cfsv2

    # 5) Match the times so that we only get station data for smallest overlapping times. 
    timestart = timestart_reanalysis if timestart_reanalysis >= timestart_cfsv2 else timestart_cfsv2
    timeend = timeend_reanalysis if timeend_reanalysis < timeend_cfsv2 else timeend_cfsv2
    timein = timestart.strftime('%Y%m%d %H:%M')
    timeout = timeend.strftime('%Y%m%d %H:%M')
    utilities.log.info('ADC provided times are {} and {}'.format(timein, timeout))

    # 6) Setup OBS specific YML-resident values
    utilities.log.info('Fetch Observations')
    #obs_yamlname = os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'obs.yml')

    # Could also set stations to None
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir)
    outfiles['OBS_DETAILED_PKL']=detailedpkl
    outfiles['OBS_SMOOTHED_PKL']=smoothedpkl
    outfiles['OBS_METADATA_PKL']=metapkl
    outfiles['OBS_NOAA_COOPS_URLS_CSV']=urlcsv
    outfiles['OBS_EXCLUDED_CSV']=exccsv
    outfiles['OBS_DETAILED_JSON']=detailedJ
    outfiles['OBS_SMOOTHED_JSON']=smoothedJ
    outfiles['OBS_METADATA_JSON']=metaJ
    utilities.log.info('Completed OBS: Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {}, DetailedJ {}, SmoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ))

    # 7) Compute DIFFS between REANALYSIS ansd CFSV2 
    utilities.log.info('Error computation NOTIDAL corrections')
    err_yamlname='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/config/err.yml'
    meta = outfiles['OBS_METADATA_PKL']
    obsf = outfiles['OBS_SMOOTHED_PKL']
    adcf = outfiles['ADCIRC_WL_PKL']
    adcR = outfiles['ADCIRC_REANALYSIS_WL_PKL']
    adcC = outfiles['ADCIRC_CFSV2_WL_PKL']

    errf, finalf, cyclef, metaf, mergedf, jsonf = exec_error(adcR, adcC, meta, err_yamlname, rootdir, iometadata, iosubdir)
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
    args = parser.parse_args()
    sys.exit(main(args))

