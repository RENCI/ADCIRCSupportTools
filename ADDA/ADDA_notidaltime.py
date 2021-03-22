###########################################################################
# ADCIRC Data Assimilation prototype pipeline. Oct 2020
# This file simply identifies some of the basic steps required
# in the final DA procedure.
# This is slightly modified touse the modified ADCIRCSupportTools methods instead
# of the original ADDA
###########################################################################

# Need large memory to run this job
import os
import sys
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# import datetime as dt
# from datetime import timedelta
# import netCDF4 as nc4
# from siphon.catalog import TDSCatalog

from utilities.utilities import utilities as utilities
from utilities import CurrentDateCycle as cdc
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from get_obs_stations.GetObsStations import GetObsStations
from compute_error_field.computeErrorField import computeErrorField
from compute_error_field.interpolateScalerField import interpolateScalerField
import visualization.diagnostics as diag


#from mpl_toolkits.basemap import Basemap
#import SRC.MakeFigs as mf

###########################################################################
# ADDA functions

def IsNewJob(iometadata, rootdir):
    finalErrorSummary = 'stationSummaryAves'+iometadata+'.csv'
    subdir = 'errorfield'
    fielddir = rootdir 
    finalErrorFilename = utilities.getSubdirectoryFileName(fielddir, subdir, finalErrorSummary)
    return not os.path.exists(finalErrorFilename)

# this is only used by main
def str2datetime(s):
    from datetime import datetime
    try:
        d = datetime.strptime(s, "%Y-%m-%d %H:%M")
    except:
        utilities.log.error("Override time2 has bad format: Should be %Y-%m-%d %H:%M".format(s))
        sys.exit('Exit over bad input time format')
    utilities.log.info("Overriding NOW as time2 specification: {}".format(d))
    return d

# noinspection PyPep8Naming,DuplicatedCode
def main(args):
    utilities.log.info(args)

    #experimentTag = 'CycleTest' # Build a directory UNDER RUNTIMEDIR Then we can iterate over progress steps
    #We anticipate lots of run swith the metaname timein-timeout (eg ADDA_202002231200_202002271200) to
    #BE found underneath 
    experimentTag = args.experiment_name
    visualiseErrorField = args.vis_error
    displayStationURLs = args.station_webpages
    vizScatterPlot = args.vis_scatter
    vizHistograms = args.error_histograms
    cvKriging = args.cv_kriging
    percentStationMissing = args.station_missing_threshold
    adcdataformat = args.adc_fortran_61
    overrideRepeats = args.override_repeats
    overridetimeout = None if args.time2 is None else str2datetime(args.time2)
    aveper = None if args.aveper is None else int(args.aveper)
    chosengrid=args.grid

    ###########################################################################
    # First Step: Load main yaml configuration information and determine the
    # 

    # 1) Setup main config data
    config = utilities.load_config()

    # Such as node_idx data
    utilities.log.info('Fetch station data for ghrid {}'.format(chosengrid))
    try:
        stationFile=config['STATIONS'][chosengrid.upper()]
        stationFile=os.path.join(os.path.dirname(__file__), "../config", stationFile)
    except KeyError as e:
        utilities.log.error('ADDA: Error specifying grid. Uppercase version Not found in the main.yml {}'.format(chosengrid))
        utilities.log.error(e)
        sys.exit()

    # Read the stationids and nodeid data
    df = pd.read_csv(stationFile, index_col=0, header=0, skiprows=[1], sep=',')
    station_id = df["stationid"].values.reshape(-1,)
    node_idx = df["Node"].values.reshape(-1,1) # had to slightly change the shaping here
    utilities.log.info('Retrived stations and nodeids from {}'.format(stationFile))

    # 3) Get ADCIRC data: Use it to decide on desired stations and time ranges
    # Build metadata string

    adc_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
    adc = Adcirc(adc_yamlname, grid=chosengrid)

    if overridetimeout is None:
        adc.set_times()  # Get current ADC now and chosen starting time
    else:
        adc.set_times(dtime2=overridetimeout.strftime('%Y-%m-%d %H:%M'))

    iometadata = '_'+adc.T1.strftime('%Y%m%d%H%M')+'_'+adc.T2.strftime('%Y%m%d%H%M') # Used for all classes downstream

    main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
    if experimentTag is None:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='ADDA'+iometadata)
    else:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='ADDA'+experimentTag+iometadata)

    ##############################################################################
    # See if job was already computed. If so skip (unless CLI override)
    # Just check for the Error summary filename for now

    status_newjob = True

    #sys.exit('IsNewJob identified this run as being previously completed time data are {}.'.format(iometadata))
        
    if overrideRepeats:
        utilities.log.info('Not a new job but Will override existence test')
        status_newjob = False 
    else:
        if IsNewJob(iometadata, rootdir):
            utilities.log.info('IsNewJob declares to be a new job')
            status_newjob = False

    if status_newjob:
        utilities.log.error('IsNewJob identified this run as being previously completed time data are {}.'.format(iometadata))
        sys.exit('IsNewJob identified this run as being previously completed time data are {}.'.format(iometadata))

    adc.get_urls()
    adc.get_grid_coords()  # populates the gridx,gridy terms
    adcirc_gridx = adc.gridx[:]
    adcirc_gridy = adc.gridy[:]

    ##############################################################################
    #config = utilities.load_config()
    ADCdir = rootdir
    ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'

    # If we were to read URLs then we would want to update the timei /out with those from levels_63
    if not os.path.exists(ADCfile):
        if adcdataformat:
            utilities.log.error('ADC: Format 61 is not implemented yet')
            sys.exit('ADC: Format 61 is not implemented yet')
        else:
            df = get_water_levels63(adc.urls, node_idx, station_id) # Gets ADCIRC water levels
            #adc.T1 = df.index[0]
            #adc.T2 = df.index[-1]
            df.to_pickle(ADCfile)
    else:
        utilities.log.info("adc_wl.pkl exists.  Using that...")
        utilities.log.info(ADCfile)
        df = pd.read_pickle(ADCfile)
        #adc.T1 = df.index[0]  # DO not reset here simply to ensure being the same as the old ADDA. 
        #adc.T2 = df.index[-1]
        utilities.log.info('read times from existing pkl {}, {}'.format(adc.T1,adc.T2))

    utilities.log.info('Done ADCIRC Reads')

    ###########################################################################
    # Get current set of water_levels from the product list class
    # NOTE: This class will double check that all the desired stations are present.
    # If not, only the ADCIRC stations are kept

    stations = df.keys()
    timein = adc.T1
    timeout = adc.T2

    # These are all defaults and can be excluded from the invocation
    obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
    obs_config = utilities.load_config(obs_yamlname)['OBSERVATIONS']
    #obs_config = obs_config['OBSERVATIONS']
    datum = obs_config['DATUM']  # 'MSL'
    unit = obs_config['UNIT']  # 'metric'
    timezone = obs_config['TIMEZONE']  # 'gmt'
    product = obs_config['PRODUCT']  # 'water_level'

    water_object = GetObsStations(datum=datum, unit=unit, product=product,timezone=timezone, iosubdir='obspkl', rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata)
    #water_object = rpl.retrieveRealProductLevels(datum=datum, unit=unit, product=product,
    #                                             timezone=timezone, metadata=iometadata, rootdir=rootdir)
    #Here we got station from adcirc but stationList may be fewer depending on nans,etc
    df_stationData, stationList = water_object.fetchStationMetaDataFromIDs(stations)

    # This should also return a new station list: but the new index may be sufficient
    # Should do this BEFORE smoothing as it will simply interpolate over the nans
    # df_pruned = water_object.removeMissingProducts(df_pruned, count_nan, percentage_cutoff=percentStationMissing)

    # Valid times that get excluded by windowing are replaced by actual values from the station obs
    # SMoother already (potentially) runs the missingness filter before interpolation

    df_pruned, count_nan, newstationlist, excludelist = water_object.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)

    ## This should also return a new station list: but the new index may be sufficient
    #df_pruned = water_object.removeMissingProducts(df_pruned, count_nan, percentage_cutoff=percentStationMissing) 

    utilities.log.info('Construct new list of times removing smoothing artifacts')
    retained_times = df_pruned.index # some got wacked during the smoothing and nan checking

    # utilities.log.info('Number of time indices that survived smoothing {}'.format(retained_times))

    # Fetch final smoothed PKL file
    # Join merged and excluded stations into a single set of URLs for subsequent interrogation

    dummy = water_object.buildURLsForStationPlotting(stationList, timein, timeout) # Need this to build urlcsv
    outputdict = water_object.writeFilesToDisk()
    obs_wl_detailed=outputdict['PKLdetailed']
    detailedJ=outputdict['JSONdetailed']
    obs_wl_smoothed=outputdict['PKLsmoothed']
    smoothedJ=outputdict['JSONsmoothed']
    metadata=outputdict['PKLmeta']
    metaJ=outputdict['JSONmeta']
    urlcsv=outputdict['CSVurl']
    exccsv=outputdict['CSVexclude']


    ###########################################################################
    # Construct error file based on the existing station data and time

    adcf = ADCfile
    obsf = obs_wl_smoothed
    meta = metadata

    err_yamlname = err_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'err.yml')
    utilities.log.info('Override aveper flag set to '+str(aveper))
    compError = computeErrorField(obsf, adcf, meta, yamlname = err_yamlname, rootdir=rootdir, aveper=aveper)
    errf, finalf, cyclef, metaf, mergedf, jsonf = compError.executePipelineNoTidalTransform( metadata = iometadata, subdir='errorfield' )
    utilities.log.info('output files '+errf+' '+finalf+' '+cyclef+' '+metaf+' '+mergedf+' '+jsonf)
    
    ###########################################################################
    # Interpolate adcirc node data using the previously generated error matrix
    # Get clamping data and prepare to merge with error file

    # Remove outliers before coming here

    ############################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--cv_kriging', action='store_true',
                        help='Boolean: Invoke a CV procedure prior to fitting kriging model')
    parser.add_argument('--station_missing_threshold', action='store', dest='station_missing_threshold', default=100,
                        help='Float: maximum percent missing station data')
    parser.add_argument('--adc_fortran_61', action='store_true',
                        help='Boolean: Choose Fortran.61 method instead of Fortran.63')
    parser.add_argument('--override_repeats',action='store_true',
                        help='Boolean: Force rerunning a pipeline even if timein-timeout flank is already done')
    parser.add_argument('--time2', action='store', dest='time2', default=None,
                        help='String: YYYY-mm-dd hh:mm : Force a value for timeout else code will use NOW')
    parser.add_argument('--aveper', action='store', dest='aveper', default=None, type=int,
                        help='int: 4 : Override number of periods for averaging')
    parser.add_argument('--vis_error', action='store_true', 
                        help='Boolean: plot on visual grid the error field')
    parser.add_argument('--vis_scatter', action='store_true', 
                        help='Boolean: plot scatterplot of station level errors')
    parser.add_argument('--station_webpages', action='store_true', 
                        help='Boolean: Fetch URLs for NOAA station levels')
    parser.add_argument('--error_histograms', action='store_true', 
                        help='Boolean: Display 3 histograms: station only, vis grid errors, and adcirc nodes')
    parser.add_argument('--grid', default='hsofs',help='Choose name of available grid',type=str)
    args = parser.parse_args()
    sys.exit(main(args))
