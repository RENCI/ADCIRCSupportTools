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
import numpy as np
import json

from utilities.utilities import utilities as utilities
from utilities import CurrentDateCycle as cdc
from get_adcirc.GetADCIRC import Adcirc, get_water_levels63
from get_obs_stations.GetObsStations import GetObsStations
from compute_error_field.computeErrorField import computeErrorField
from compute_error_field.interpolateScalerField import interpolateScalerField
import visualization.diagnostics as diag

import time as tm


###########################################################################

import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

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

    ##
    ## Start the timing
    ##

    timedata = list()
    for itimes in range(0,10):
        t0 = tm.time()

        obs_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
        obs_config = utilities.load_config(obs_yamlname)
        station_df = utilities.get_station_list()
        station_id = station_df["stationid"].values.reshape(-1,)
        node_idx = station_df["Node"].values
    
        adc_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
        adc = Adcirc(adc_yamlname)
        if overridetimeout is None:
            adc.set_times()  # Get current ADC now and chosen starting time
        else:
            adc.set_times(dtime2=overridetimeout.strftime('%Y-%m-%d %H'))
        iometadata = '_'+adc.T1.strftime('%Y%m%d%H%M')+'_'+adc.T2.strftime('%Y%m%d%H%M') # Used for all classes downstream
        main_config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
        if experimentTag is None:
            rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='ADDA'+iometadata+'_'+str(itimes))
        else:
            rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='ADDA'+experimentTag+iometadata)
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
        adc.get_urls()
        adc.get_grid_coords()  # populates the gridx,gridy terms
        adcirc_gridx = adc.gridx[:]
        adcirc_gridy = adc.gridy[:]
    
        ADCdir = rootdir
        ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'
        if not os.path.exists(ADCfile):
            if adcdataformat:
                utilities.log.error('ADC: Format 61 is not implemented yet')
                sys.exit('ADC: Format 61 is not implemented yet')
            else:
                df = get_water_levels63(adc.urls, node_idx, station_id) # Gets ADCIRC water levels
                df.to_pickle(ADCfile)
        else:
            utilities.log.info("adc_wl.pkl exists.  Using that...")
            utilities.log.info(ADCfile)
            df = pd.read_pickle(ADCfile)
            utilities.log.info('read times from existing pkl {}, {}'.format(adc.T1,adc.T2))
        utilities.log.info('Done ADCIRC Reads')
        stations = df.keys()
        timein = adc.T1
        timeout = adc.T2
        obs_config = obs_config['OBSERVATIONS']
        datum = obs_config['DATUM']  # 'MSL'
        unit = obs_config['UNIT']  # 'metric'
        timezone = obs_config['TIMEZONE']  # 'gmt'
        product = obs_config['PRODUCT']  # 'water_level'
    
        water_object = GetObsStations(datum=datum, unit=unit, product=product,timezone=timezone, iosubdir='obspkl', rootdir=rootdir, yamlname=obs_yamlname, metadata=iometadata)
        df_stationData, stationList = water_object.fetchStationMetaDataFromIDs(stations)
        df_pruned, count_nan, newstationlist, excludelist = water_object.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
        utilities.log.info('Construct new list of times removing smoothing artifacts')
        retained_times = df_pruned.index # some got wacked during the smoothing and nan checking
        dummy = water_object.writeURLsForStationPlotting(stationList, timein, timeout) # Need this to build urlcsv
        obs_wl_detailed, obs_wl_smoothed, metadata, urlcsv, exccsv, metaJ, detailedJ, smoothedJ = water_object.fetchOutputNames()
    
        adcf = ADCfile
        obsf = obs_wl_smoothed
        meta = metadata
        err_yamlname = err_yamlname = os.path.join(os.path.dirname(__file__), '../config', 'err.yml')
        utilities.log.info('Override aveper flag set to '+str(aveper))
        compError = computeErrorField(obsf, adcf, meta, yamlname = err_yamlname, rootdir=rootdir, aveper=aveper)
        errf, finalf, cyclef, metaf, mergedf, jsonf = compError.executePipeline( metadata = iometadata, subdir='errorfield' )
        utilities.log.info('output files '+errf+' '+finalf+' '+cyclef+' '+metaf+' '+mergedf+' '+jsonf)

        inerrorfile = finalf
        int_yamlname=os.path.join(os.path.dirname(__file__), '../config', 'int.yml')
        #int_config = utilities.load_config(int_yamlname)
        #clampfile=os.path.join(os.path.dirname(__file__), "../config", int_config['DEFAULT']['ClampList'])
        #clampfile='/home/jtilson/ADCIRCSupportTools/config/clamp_list_hsofs.dat'
        # If clampingfile is None then it will be detemrioned from the inpout config
        #krig_object = interpolateScalerField(datafile=inerrorfile, yamlname=int_yamlname, clampingfile=None, metadata=iometadata, rootdir=rootdir)
        krig_object = interpolateScalerField(datafile=inerrorfile, yamlname=int_yamlname, metadata=iometadata, rootdir=rootdir)
        extraFilebit=''
    
        vparams=None    # If these are none then singleStep would simply read the yaml
        param_dict=None
    
        utilities.log.info('doing a single krige using current best parameters')
        utilities.log.info('param_dict: {}'.format(param_dict))
        utilities.log.info('vparams: {}'.format(vparams))
        model_filename = 'interpolate_model'+extraFilebit+iometadata+'.h5' if cvKriging else 'interpolate_model'+iometadata+'.h5'
    
        status = krig_object.singleStepKrigingFit( param_dict, vparams, filename = model_filename)
        gridx, gridy = krig_object.input_grid() # Grab from the config file
        df_grid = krig_object.krigingTransform(gridx, gridy,style='grid',filename = model_filename)
        gridz = df_grid['value'].values
        n=gridx.shape[0]
        gridz = gridz.reshape(-1, n)
        krig_object.plot_model(gridx, gridy, gridz, keepfile=True, filename='image'+iometadata+'.png', metadata=iometadata)
        df_adcirc_grid = krig_object.krigingTransform(adcirc_gridx, adcirc_gridy, style='points', filename = model_filename)
        krig_adcircfilename = krig_object.writeADCIRCFormattedTransformedDataToDisk(df_adcirc_grid)
        krig_interfilename = krig_object.writeTransformedDataToDisk(df_grid)
        utilities.log.info('Transformed interpolated data are in '+krig_interfilename)
        utilities.log.info('Transformed interpolated ADCIRC formatteddata are in '+krig_adcircfilename)
        newx, newy, newz = krig_object.fetchRawInputData()  # Returns the unclamped input data for use by the scatter method
        timedata.append( tm.time()-t0 )

    m, mmh,mph = mean_confidence_interval(timedata)
    print('Total times: Mean {} 95% CI range {}-{}'.format(m, mmh, mph))

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
    args = parser.parse_args()
    sys.exit(main(args))
