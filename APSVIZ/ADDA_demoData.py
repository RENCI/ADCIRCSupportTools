###########################################################################
# ADCIRC Data Assimilation prototype pipeline. Jan 2020
# This file simply identifies some of the basic steps required
# in the final DA procedure.

# Modification: We only want the ADC and OBS data no errrors for an upcoming demo

###########################################################################

# Need large memory to run this job
import os
import sys
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities.utilities import utilities as utilities
from SRC.get_adcirc import AdcircDemo , get_water_levels63
import SRC.retrieveRealProductLevels as rpl
from mpl_toolkits.basemap import Basemap

###########################################################################
# ADDA functions

def IsNewJob(iometadata, rootdir):
    finalErrorSummary = 'stationSummaryAves_'+iometadata+'.csv'
    subdir = subdir = 'errorfield'
    fielddir = rootdir 
    finalErrorFilename = utilities.getSubdirectoryFileName(fielddir, subdir, finalErrorSummary)
    return not os.path.exists(finalErrorFilename)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2datetime(s):
    from datetime import datetime
    try:
        d = datetime.strptime(s, "%Y%m%d-%H:%M:%S")
    except:
        utilities.log.error("Override time2 has bad format:".format(s))
    utilities.log.info("Overriding NOW as time2 specification: {}".format(d))
    return d

# noinspection PyPep8Naming,DuplicatedCode
def main(args):
    utilities.log.info(args)

    experimentTag = args.experiment_name

    visualiseErrorField = args.vis_error
    displayStationURLs = args.station_webpages
    cvKriging = args.cv_kriging
    percentStationMissing = args.station_missing_threshold
    adcdataformat = args.adc_fortran_63
    vizScatterPlot = args.vis_scatter
    vizHistograms = args.error_histograms
    overrideRepeats = args.override_repeats
    overridetimeout = None if args.time2==None else str2datetime(args.time2)
    aveper = None if args.aveper==None else int(args.aveper)

    ###########################################################################
    # First Step: Load current yaml configuration information and determine the
    # current time range to perform the processing

    config = utilities.load_config()

    ###########################################################################
    # Get ADCIRC data: Use it to decide on desired stations and time ranges
    # Build proper metadata string

    adc = Adcirc()

    # Get from productlevel class instead?
    station_df = utilities.get_station_list()
    station_id = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values

    if overridetimeout==None:
        adc.set_times() # Get current ADC now and chosen starting time
    else:
        adc.set_times(overridetimeout.strftime('%Y-%m-%d %H'))

    adc.get_urls()
    urls = adc.urls

    adc.get_grid_coords()  # populates the gridx,gridy terms
    adcirc_gridx = adc.gridx[:]
    adcirc_gridy = adc.gridy[:]

    # fetch RUNTIMEDIR directory: Build metadata string used by all subsequent methods

    iometadata = adc.T1.strftime('%Y%m%d%H%M')+'_'+adc.T2.strftime('%Y%m%d%H%M') # Used for all classes downstream

    if experimentTag==None:
        rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADDA_'+iometadata)
    else:
        rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='ADDA_'+experimentTag+'_'+iometadata)

    ##############################################################################
    # See if job was already computed. If so skip (unless CLI override)
    # Just check for the Error summary filename for now

    status_newjob = True

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

    ##############################################################################

    config = utilities.load_config()
    ADCdir = rootdir
    ADCfile = ADCdir+'/adc_wl_'+iometadata+'.pkl'

    if not os.path.exists(ADCfile):
        if adcdataformat:
            df = get_water_levels63(adc.urls, node_idx, station_id) # Gets ADCIRC water levels
            df.to_pickle(ADCfile)
        else:
            sys.exit('ADC: Format 61 is not implemented yet')
    else:
        utilities.log.info("adc_wl.pkl exists.  Using that...")
        utilities.log.info(ADCfile)
        df = pd.read_pickle(ADCfile)

    utilities.log.info('Done ADCIRC Reads')

    ###########################################################################
    # Get current set of water_levels from the product list class
    # NOTE: This class will double check that all the desired stations are present.
    # If not, only the ADCIRC stations are kept

    stations = df.keys()
    timein = adc.T1
    timeout = adc.T2

    # These are all defaults and can be excluded from the invocation
    obs_config = config['OBSERVATIONS']
    datum = obs_config['DATUM']  # 'MSL'
    unit = obs_config['UNIT']  # 'metric'
    timezone = obs_config['TIMEZONE']  # 'gmt'
    product = obs_config['PRODUCT']  # 'water_level'
    water_object = rpl.retrieveRealProductLevels(datum=datum, unit=unit, product=product,
                                                 timezone=timezone, metadata=iometadata, rootdir=rootdir)

    #Here we got station from adcirc but stationList may be fewer depending on nans,etc
    df_stationData, stationList = water_object.fetchStationMetaDataFromIDs(stations)

    # This should also return a new station list: but the new index may be suffucient
    # SHould do this BEFORE smotthing as it will simply interpolate over the nans
    # df_pruned = water_object.removeMissingProducts(df_pruned, count_nan, percentage_cutoff=percentStationMissing)

    # Valid times that get excluded by windowing are replaced by actual values from the station obs
    # SMoother already (potentially) runs the missingness filter before interpolation

    df_pruned, count_nan, newstationlist = water_object.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)

    ## This should also return a new station list: but the new index may be suffucient
    #df_pruned = water_object.removeMissingProducts(df_pruned, count_nan, percentage_cutoff=percentStationMissing) 

    utilities.log.info('Construct new list of times removing smoothing artifacts')
    retained_times = df_pruned.index # some got wacked during the smoothing and nan checking

    utilities.log.info('Number of time indices that survived smoothing {}'.format(retained_times))

    # Fetch final smoothed PKL file
    # Join merged and excluded stations into a single set of URLs for subsequent interrogation

    dummy = water_object.buildURLsForStationPlotting(stationList, timein, timeout, showWeb=False) # Need this to build urlcsv
    obs_wl_detailed, obs_wl_smoothed, metadata, urlcsv, exccsv = water_object.fetchOutputNames()

    ###########################################################################
    # Construct error file based on the existing station data and time

    adcf = ADCfile
    obsf = obs_wl_smoothed
    meta = metadata

    utilities.log.info('Override aveper flag set to '+str(aveper))
    compError = cmp.computeErrorField(obsf, adcf, meta, rootdir=rootdir, aveper=aveper)
    errf, finalf, cyclef, metaf, mergedf = compError.executePipeline( metadata = iometadata )
    utilities.log.info('output files '+errf+' '+finalf+' '+cyclef+' '+metaf+' '+mergedf)
    
    ###########################################################################
    # Interpolate adcirc node data using the previously generated error matrix
    # Get clamping data and prepare to merge with error file

    # Remove outliers before coming here

    inerrorfile = finalf
    clampfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ClampList'])

    krig_object = isf.interpolateScalerField(inerrorfile, clampfile, metadata=iometadata, rootdir=rootdir)

    if cvKriging:
        utilities.log.info('Building kriging model using CV procedure')
        status = krig_object.CVKrigingFit(filename='interpolate_model_'+iometadata+'.h5' )
    else:
        status = krig_object.singleStepKrigingFit(filename = 'interpolate_model_'+iometadata+'.h5')

    # Use a simple grid for generating visualization work
    gridx, gridy = krig_object.input_grid() # Grab from the config file
    df_grid = krig_object.krigingTransform(gridx, gridy,style='grid',filename = 'interpolate_model_'+iometadata+'.h5')
    # Pass dataframe for the plotter
    gridz = df_grid['value'].values
    n=gridx.shape[0]
    gridz = gridz.reshape(-1, n)
    krig_object.plot_model(gridx, gridy, gridz, keepfile=True, filename='image_'+iometadata+'.png', metadata=iometadata)

    # Repeat now using the real adcirc data adcirc_gridx, and adcirc_gridy 
    df_adcirc_grid = krig_object.krigingTransform(adcirc_gridx, adcirc_gridy, style='points', filename='interpolate_model_'+iometadata+'.h5')

    krig_adcircfilename = krig_object.writeADCIRCFormattedTransformedDataToDisk(df_adcirc_grid)
    krig_interfilename = krig_object.writeTransformedDataToDisk(df_grid)

    utilities.log.info('Transformed interpolated data are in '+krig_interfilename)
    utilities.log.info('Transformed interpolated ADCIRC formatteddata are in '+krig_adcircfilename)

    # Fetch non clamped input (station-level) data for scatter plotting
    newx, newy, newz = krig_object.fetchRawInputData()  # Returns the unclamped input data for use by the scatter method

    ########################################################################
    # Perform some possible diagnostics

    krig_object.plot_scatter_discrete(newx,newy,newz,showfile=False, keepfile=True, filename='image_Discrete_'+iometadata+'.png', metadata='testmetadataDiscrete')

    # Get error statistics for both the visualization and actual ADC IRC nodes data

    if vizHistograms:
        vmax = config['GRAPHICS']['VMAX']
        vmin = config['GRAPHICS']['VMIN']
        # Actual station-level (summary) (mean) differences
        print('Station level mean errors')
        df_test = pd.read_csv(inerrorfile)
        df_test['mean'].hist(range=[vmin,vmax])
        plt.show()
        df_test['mean'].max()
        df_test['mean'].min()
        df_test['mean'].count()
        df_test['mean'].mean()
        print('Station level STD errors')
        # Actual station-level (summary) (std) differences
        df_test = pd.read_csv(inerrorfile)
        df_test['std'].hist(range=[vmin,vmax])
        plt.show()
        df_test['std'].max()
        df_test['std'].min()
        df_test['std'].count()
        df_test['std'].mean()
        # Visualization grid (2D data)
        print('Visual grid mean errors')
        df_test = pd.read_pickle(krig_interfilename)
        df_test['value'].hist(range=[vmin,vmax])
        plt.show()
        df_test['value'].max()
        df_test['value'].min()
        df_test['value'].count()
        df_test['value'].mean()
        # ADCIRC interpolated data (1D data)
        print('ADCIRC node list mean errors')
        df_test = pd.read_csv(krig_adcircfilename, skiprows=3)
        df_test.columns = ('node','value')
        df_test['value'].hist(range=[vmin,vmax])
        plt.show()
        df_test['value'].max()
        df_test['value'].min()
        df_test['value'].count()
        df_test['value'].mean()

    ##########################################################################
    # Test diagnostics
    # Station level plots

    if displayStationURLs:
        urlfilename = urlcsv
        diag.displayURLsForStations(urlfilename, stationList)

    if visualiseErrorField:
        print('plot new diagnostics')
        #selfX,selfY,selfZ = krig_object.fetchRawInputData() # Returns the unclamped input data for use by the scatter method
        selfX, selfY, selfZ = krig_object.fetchInputAndClamp()  # Returns the clamped input data for use by the scatter method
        diag.plot_interpolation_model(gridx, gridy, gridz, selfX, selfY, selfZ, metadata=iometadata)

    if vizScatterPlot:
        print('final scatter plot')
        diag.plot_scatter_discrete(newx, newy, newz, metadata=iometadata)

    ############################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--vis_error', type=str2bool, action='store', dest='vis_error', default=False,
                        help='Boolean: plot on visual grid the error field')
    parser.add_argument('--vis_scatter', type=str2bool, action='store', dest='vis_scatter', default=True,
                        help='Boolean: plot scatterplot of station level errors')
    parser.add_argument('--station_webpages', type=str2bool, action='store', dest='station_webpages', default=False,
                        help='Boolean: Fetch URLs for NOAA station levels')
    parser.add_argument('--cv_kriging', type=str2bool, action='store', dest='cv_kriging', default=False,
                        help='Boolean: Invoke a CV procedure prior to fitting kriging model')
    parser.add_argument('--station_missing_threshold', action='store', dest='station_missing_threshold', default=100,
                        help='Float: maximum percent missing station data')
    parser.add_argument('--adc_fortran_63', type=str2bool, action='store', dest='adc_fortran_63', default=True,
                        help='Boolean: Choose either Fortran.63 method or Fortran.61 method')
    parser.add_argument('--error_histograms', type=str2bool, action='store', dest='error_histograms', default=True,
                        help='Boolean: Display histograms station only, vis grid errors, and adcirc nodes')
    parser.add_argument('--override_repeats', type=str2bool, action='store', dest='override_repeats', default=False,
                        help='Boolean: Force rerunning a pipeline even if timein-timeout flank is already done')
    parser.add_argument('--time2', action='store', dest='time2', default=None,
                        help='String: YYYYmmdd-hh:mm:ss : Force a value for timeout else code will use NOW')
    parser.add_argument('--aveper', action='store', dest='aveper', default=None, type=int,
                        help='int: 4 : Override number of periods for averaging')
    args = parser.parse_args()
    sys.exit(main(args))
