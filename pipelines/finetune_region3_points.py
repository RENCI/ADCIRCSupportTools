#!/usr/bin/env python

###################################################################
##
## Pipeline that is useful in the APSVIZ context. Entry an adcirc URL and the nodelist
## and generate the ADS,OBS,ERR data sets.
##
## Read the input url forecast first, so that we can get its timein, and used that to 
## specify the timeout for a previous nowcast.
###################################################################

import os,sys
import shutil
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

## Little trick 
## For now this sidesteps the differing nomenclatures used for 2020 and 2021 data storage
## Moving forward this needs to be handled much more generally 

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

def exec_adcirc(dtime2, rootdir, iometadata, adc_yamlname, node_idx, station_ids, grid, doffset=-4):
    """
    dtime2 arrives as a string in the format YYYY-mm-dd MM:SS 
    This step is a little tricky for multiple grids. Because the "time semantics" tells the code
    to construct URLS based on parameters in the json. SO we need to intercept some pof those params and change
    them if the input grid is not hsofs.
    """
    # Start the fetch of ADCIRC data
    adc = Adcirc(adc_yamlname, grid=grid)
    adc.set_times(dtime2=dtime2, doffset=doffset) # Global time get set for use by get_urls
    utilities.log.info("T1 (start) = {}".format(adc.T1))
    utilities.log.info("T2 (end)   = {}".format(adc.T2))

    # Check the year and update Instance if 2021
    newconfig = buildLocalConfig()
    adc.get_urls(inconfig=newconfig)
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

def exec_adcirc_forecast(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids, grid):
    """
    This step is always passed actual URLs to fetch the forecast data. So changing grid names here is easy
    """
    adc = Adcirc(adc_yamlname, grid=grid)
    adc.urls = urls
    utilities.log.info("List of available urls input specification:")
    ## Meaningless utilities.log.info('Observed TIMES are T1 {}, T2 {}'.format(adc.T1.strftime('%Y%m%d%H%M'), adc.T2.strftime('%Y%m%d%H%M')))
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl_forecast',iometadata=iometadata)
    ##df.to_pickle(ADCfile)
    ####df.to_json(ADCjson)
    print('write new json')
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl_forecast')
    #timestart = adc.T1.strftime('%Y%m%d%H%M')
    #timeend = adc.T2.strftime('%Y%m%d%H%M')
    timestart = adc.T1
    timeend = adc.T2
    df.index = pd.to_datetime(df.index)
    rundate = generateRUNTIMEmetadata(df)
    return ADCfile, ADCjson, timestart, timeend, rundate

def exec_adcirc_nowcast(inurls, rootdir, iometadata, adc_yamlname, node_idx, station_ids, grid, doffset=-4):
    """
    This step is always passed an actual forecast URL. nowcast urs are built from it
    """
    adc = Adcirc(adc_yamlname, grid=grid)
    adc.get_urls_noyaml(inurls, doffset) # doffset in -days internally will be converted to 6hours periods (+1)
    #adc.urls = nc_urls
    utilities.log.info("New: List of available urls input specification:")
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl_forecast',iometadata=iometadata)
    ##df.to_pickle(ADCfile)
    ####df.to_json(ADCjson)
    print('write new json')
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl_forecast')
    #timestart = adc.T1.strftime('%Y%m%d%H%M')
    #timeend = adc.T2.strftime('%Y%m%d%H%M')
    timestart = adc.T1
    timeend = adc.T2
    df.index = pd.to_datetime(df.index)
    rundate = generateRUNTIMEmetadata(df)
    return ADCfile, ADCjson, timestart, timeend

def exec_adcirc_nowcast_hurricane(inurls, rootdir, iometadata, adc_yamlname, node_idx, station_ids, grid, doffset=-4):
    """
    This step is always passed an actual forecast URL. nowcast urs are built from it
    """
    adc = Adcirc(adc_yamlname, grid=grid)
    adc.get_urls_hurricane_noyaml(inurls, doffset) # doffset in -days internally will be converted to 6hours periods (+1)
    #adc.urls = nc_urls
    utilities.log.info("New: List of available hurricane urls input specification:")
    ADCfile = rootdir+'/adc_wl'+iometadata+'.pkl'
    ADCjson = rootdir+'/adc_wl'+iometadata+'.json'
    df = get_water_levels63(adc.urls, node_idx, station_ids) # Gets ADCIRC water levels
    adc.T1 = df.index[0] # Optional update to actual times fetched form ADC
    adc.T2 = df.index[-1]
    ADCfile = utilities.writePickle(df, rootdir=rootdir,subdir='',fileroot='adc_wl_forecast',iometadata=iometadata)
    ##df.to_pickle(ADCfile)
    ####df.to_json(ADCjson)
    print('write new json')
    ADCjson=writeToJSON(df, rootdir, iometadata,fileroot='adc_wl_forecast')
    #timestart = adc.T1.strftime('%Y%m%d%H%M')
    #timeend = adc.T2.strftime('%Y%m%d%H%M')
    timestart = adc.T1
    timeend = adc.T2
    df.index = pd.to_datetime(df.index)
    rundate = generateRUNTIMEmetadata(df)
    return ADCfile, ADCjson, timestart, timeend


def exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir, stationFile):
    rpl = GetObsStations(iosubdir=iosubdir, rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata, stationFile=stationFile)
    df_stationNodelist = rpl.fetchStationNodeList()
    stations = df_stationNodelist['stationid'].to_list()
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

def buildNowcast(urls):
    """
    Take the input dict, grab the url and convert from a ASGS forecast to a nowcast
    Special cases suc h as hurricane events are not yet handled
    I dont know how many urls may present, so we assume all must be converted.
    """ 
    for key,url in urls.items():
        print('key{} url{}'.format(key,url))
    return urls
    
def buildLocalConfig(grid='hsofs'):
    """
    Temporary function to build a config object that can be passed to an ADCIRC nowcast
    We need to beable to update Instance and year
    We need to insert, temporarily, some code to account for an alternative grid
    """
    cfg = dict() 
    if grid=='hsofs':
        cfg['AdcircGrid']= "hsofs"
        cfg['Machine']= "hatteras.renci.org"
        cfg['Instance']= "%s"
        cfg['baseurl']= "http://tds.renci.org:8080/thredds/"
        cfg['catPart']= "/catalog/%s/nam/catalog.xml"
        cfg['dodsCpart']= "/dodsC/%s/nam/%s/%s/%s/%s/nowcast/fort.%s.nc"
        cfg['fortNumber']= "63"
    elif grid=='ec95d':
        cfg['AdcircGrid']= "ec95d"
        cfg['Machine']= "hatteras.renci.org"
        cfg['Instance']= "%s"
        cfg['baseurl']= "http://tds.renci.org:8080/thredds/"
        cfg['catPart']= "/catalog/%s/nam/catalog.xml"
        cfg['dodsCpart']= "/dodsC/%s/nam/%s/%s/%s/%s/nowcast/fort.%s.nc"
        cfg['fortNumber']= "63" 
    else:
        utilities.log.error('Only hsofs and ec95d grids allowed at this time. {}'.format(grid))
        sys.exit()
    return cfg

def generateRUNTIMEmetadata(df):
    """ 
    Look for smallest diff and subtract one off from the lowest value
    """
    timestart = df.index.min()
    res = (pd.Series(df.index[1:]) - pd.Series(df.index[:-1])).value_counts()
    timeinc = res.index.min()
    timestartMinusIncrement = timestart - timeinc
    return timestartMinusIncrement.strftime('%Y%m%d%H')

def buildCSV(dataDict):
    """
    For APSVIZ convenience, construcyt an alternativer format of the 
    png data: Namely, 
    StationId,StationName,Lat,Lon,Node,Filename
    No quotes around Lat and Lon.
    This method is highly specific to the APSVIZ work
    """
    df=pd.DataFrame(dataDict['STATIONS']).T
    df.columns=('Lat','Lon','Node','State','StationName','Filename')
    df.index.name='StationId'
    df['Lat'] = df['Lat'].astype(float)
    df['Lon'] = df['Lon'].astype(float)
    return df[['StationName','State','Lat','Lon','Node','Filename']]

# noinspection PyPep8Naming,DuplicatedCode
def main(args):

    t0 = tm.time()
    outfiles = dict()

    doffset = args.doffset
    chosengrid=args.grid
    utilities.log.info('Grab the first 4 months of ADCIRC values at the stations skip forewcasting')
    utilities.log.info('Grid specified was {}'.format(chosengrid))

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
    elif args.inputURL != None:
        # If here we still need to build a dict for ADCIRC
        url = args.inputURL
        #dte = extractDateFromURL(url)
        dte='placeHolder' # The times will be determined from the real data
        urls={dte:url}
        utilities.log.info('Explicit URL provided {}'.format(urls))
    else:
        utilities.log.error('No Proper URL specified')

    # 0) URL nowcast timeout from it
    # NOTE the dict key (datecycle) is not actually used in this code as it is yet to be defined to me
    # We already expect this to be a single url

    # 1) Setup main config data - we will override elements later
    iosubdir = args.iosubdir
    iometadata = args.iometadata
    main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc

    if args.outputDir is None:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
    else:
        rootdir = args.outputDir
        rootdir = utilities.setBasedir(args.outputDir)
    utilities.log.info('Specified rootdir underwhich all files will be stored. Rootdir is {}'.format(rootdir))

    outfiles['OBS_CREATIONTIME']=dt.datetime.now().strftime('%Y%m%d%H%M')
    outfiles['CLUSTER_OUTPUTDIR']=rootdir
    outfiles['IOSUBDIR']=iosubdir
    outfiles['IOMETADATA']=iometadata
   
    # FETCH ADCIRC NODEIDs and station_IDs
    # Such as node_idx data required for the ADCIRC calls

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

    # 3) Setup ADCIRC specific YML-resident inputs
    utilities.log.info('Fetch ADCIRC-region3 reanalysis NOWCAST')
    adc_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')

    # Need to rename this as it is not really a forecast. it is whatever the URL says it is.

    # 4) Get actual ASGS Forecast data from this we can ghet the times
    # Not any need to specify a diff yml since we pass in the url directly
    # This will be appended to the DIFF plots in the final PNGs

    ADCfile, ADCjson, timestart, timeend, runDataMetadata = exec_adcirc_forecast(urls, rootdir, iometadata, adc_yamlname, node_idx, station_ids, chosengrid)
    utilities.log.info('Completed ADCIRC nowcast Reads')
    outfiles['ADCIRC_WL_PKL']=os.path.basename(ADCfile)
    outfiles['ADCIRC_WL_JSON']=os.path.basename(ADCjson)

    # 3) Setup OBS specific YML-resident values
    utilities.log.info('Fetch Observations')
    #obs_yamlname = os.path.join('~/ADCIRCSupportTools', 'config', 'obs.yml')

    # Grab time Range and tentative station list from the ADCIRC fetch  (stations may still be filtered out)
    timein = timestart.strftime('%Y%m%d %H:%M')
    timeout = timeend.strftime('%Y%m%d %H:%M')

    utilities.log.info('Manually limit times to Jan 01 - May 01')
    timein=timestart.strftime('20180101 00:00')
    timeout=timestart.strftime('20180501 00:00')
    utilities.log.info('Manually limit times to Jan 01 - May 01')

    utilities.log.info('ADC times are {} and {}'.format(timein, timeout))

    # New approach to account for the changing grid names
    try:
        stationFile=main_config['STATIONS'][args.grid.upper()]
    except KeyError as e:
        utilities.log.error('Error specifying grid. Uppercase version Not found in the main.yml {}'.format(args.grid))
        utilities.log.error(e)
    obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
    obs_config = utilities.load_config(obs_yamlname)
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = exec_observables(timein, timeout, obs_yamlname, rootdir, iometadata, iosubdir, stationFile)
    outfiles['OBS_GRID']=args.grid
    outfiles['OBS_STATIONFILE']=stationFile
    outfiles['OBS_DETAILED_PKL']=os.path.basename(detailedpkl)
    outfiles['OBS_SMOOTHED_PKL']=os.path.basename(smoothedpkl)
    outfiles['OBS_METADATA_PKL']=os.path.basename(metapkl)
    outfiles['OBS_NOAA_COOPS_URLS_CSV']=os.path.basename(urlcsv)
    outfiles['OBS_EXCLUDED_CSV']=os.path.basename(exccsv)
    outfiles['OBS_DETAILED_JSON']=os.path.basename(detailedJ)
    outfiles['OBS_SMOOTHED_JSON']=os.path.basename(smoothedJ)
    outfiles['OBS_METADATA_JSON']=os.path.basename(metaJ)
    utilities.log.info('Completed OBS: Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} MetaJ {}, DetailedJ {}, SmoothedJ {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv,metaJ, detailedJ, smoothedJ))

    # 4) Setup ERR specific YML-resident values
    utilities.log.info('Error computation NOTIDAL corrections')
    err_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'err.yml')
    meta = metapkl
    obsf = smoothedpkl
    adcf = ADCfile 
    errf, finalf, cyclef, metaf, mergedf, jsonf = exec_error(obsf, adcf, meta, err_yamlname, rootdir, iometadata, iosubdir)
    outfiles['ERR_TIME_PKL']=os.path.basename(errf)
    outfiles['ERR_TIME_JSON']=os.path.basename(jsonf)
    outfiles['ERR_STATION_AVES_CSV']=os.path.basename(errf)  # THis would pass to interpolator
    outfiles['ERR_STATION_PERIOD_AVES_CSV']=os.path.basename(cyclef)
    outfiles['ERR_METADATA_CSV']=os.path.basename(metaf)
    outfiles['ERR_ADCOBSERR_MERGED_CSV']=os.path.basename(mergedf) # This is useful for visualization insets of station bahavior
    utilities.log.info('Completed ERR')


    # 6) Build a series of station-PNGs.
    # Build input dict for the plotting
    files=dict()
    files['META']=metaJ # outfiles['OBS_METADATA_JSON']
    files['DIFFS']=jsonf # outfiles['ERR_TIME_JSON']
    files['FORECAST']=ADCjson # outfiles['ADCIRC_WL_FORECAST_JSON']
    utilities.log.info('PNG plotter dict is {}'.format(files))
    png_dict = exec_pngs(files=files, rootdir=rootdir, iometadata=iometadata, iosubdir=iosubdir)

    # 6b) Build a CSV version of png_dict special for the APSVIZ work
    df_png_csv = buildCSV(png_dict)
    outfilecsv = utilities.writeCsv(df_png_csv, rootdir=rootdir,subdir=iosubdir,fileroot='stationProps',iometadata='')  
    utilities.log.info('Wrote pipeline StationMetadata to {}'.format(outfilecsv))
    files['STATIONMETADATA']=outfilecsv

    # Merge dict from plotter and finish up
    outfiles.update(png_dict)
    outfilesjson = utilities.writeDictToJson(outfiles, rootdir=rootdir,subdir=iosubdir,fileroot='runProps',iometadata='') # Never change fname
    utilities.log.info('Wrote pipeline Dict data to {}'.format(outfilesjson)) 

    utilities.log.info('Finished pipeline in {} s'.format(tm.time()-t0))
    utilities.log.info('Copied log file to {}'.format('/'.join([rootdir,'logs'])))
    shutil.copy(utilities.LogFile,'/'.join([rootdir,'logs']))

    print(outfiles)
    return 0

    # Setup for computing the station diffs (aka adcirc - obs errors)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--outputDir', action='store', dest='outputDir', default=None,
                        help='Available high leverl directory')
    parser.add_argument('--ignore_pkl', help="Ignore existing pickle files.", action='store_true')
    parser.add_argument('--doffset', default=-2, help='Day lag or datetime string for analysis: def to YML -2', type=int)
    parser.add_argument('--iometadata', action='store', dest='iometadata',default='', help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default='', help='Used to locate output files into subdir', type=str)
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    parser.add_argument('--inputURL', action='store', dest='inputURL', default=None,
                        help='String: url.')
    parser.add_argument('--grid', default='hsofs',help='Choose name of available grid',type=str)
    args = parser.parse_args()
    sys.exit(main(args))

