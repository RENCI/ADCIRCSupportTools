#!/usr/bin/env python


# Invoke the interpolationm pipeline and create ADCIRC formatted files
# NOTE: there is an implied order here. We expect the input data to be lon x lat
# We maintain this order when performing interpolations/predictions
#
# This code simply interpolates a single input data set
###########################################################################

## Added code to capture the set of scores for subsequenct analysis

# Need large memory to run this job
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random 
import json
import datetime
from utilities.utilities import utilities
from compute_error_field.interpolateScalerField import interpolateScalerField
import visualization.diagnostics as diag
import seaborn as sns

###########################################################################
# For 2x2 histrogram plots
def genSinglePlot(i, fig, df_data,vmin,vmax,inputMetadata):
    ax = fig.add_subplot(2, 2, i)
    sns.distplot(df_data )
    ax.set_title(inputMetadata, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('tight')

def parseWeeklyDateFilename(infilename):
    """
    filename must be of the form stationSummaryAves_01_201801010000_201801070000.csv
    """
    utilities.log.info('Using WEEKLY form of filenames')
    words=(infilename.split('.')[-2]).split('_') 
    metadata = '_'+words[-2]+'_'+words[-1]
    return metadata
    
def parseDateFilename(infilename):
    """
    filename must be of the form stationSummaryAves_18-332_2018112800.csv
    """
    utilities.log.info('Using DAILY form of filenames')
    words=(infilename.split('.')[-2]).split('_')
    metadata = '_'+words[-2]+'_'+words[-1]
    return metadata

# New methodsw to support stratified splits
# OrderedDicts ?
def fetch_data_metadata(metaFile=None):
    metaDict=None
    try:
        #meta='/'.join([metaFile,'obs_water_level_metadata.json'])
        with open(meta, 'r') as fp:
            try:
                metaDict = json.load(fp)
            except OSError:
                utilities.log.error("Could not open/read file {}".format(meta))
                sys.exit()
    except TypeError:
        utilities.log.info('No directory specified. Set metadata filename to None')
    return metaDict

def main(args):
    utilities.log.info(args)

    cv_testing = args.cv_testing
    classdataFile = args.classdataFile
    utilities.log.info('Input classdata file is {}'.format(classdataFile))

    ###########################################################################
    # Interpolate adcirc node data using the previously generated error matrix
    # Get clamping data and prepare to merge with error file
    # Remove outliers within the compute error codes before coming here

    config = utilities.load_config()
    extraExpDir=args.subdir_name
    if args.outroot is None:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
    else:
        rootdir = args.outroot
    utilities.log.info('Specified rootdir underwhich all files will be stored. Rootdir is {}'.format(rootdir))
    ##rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'],basedirExtra=extraExpDir)
   
    # Set up interpolation YML. To find clamp file if not specified
    #utilities.log.info('Use Standard ADDA krig parameters')
    #yamlname=os.path.join(os.path.dirname(__file__), '../ADCIRCSupportTools/config', 'int.yml')
    utilities.log.info('Use longer range terms: Same as the OceanRise work')
    #yamlname=os.path.join(os.path.dirname(__file__), '../ADCIRCSupportTools/config', 'int.REANALYSIS.yml ')

    #if args.inrange is not None:
    #    config=buildLocalConfig(n_range=args.inrange)
    #else:

    if args.yamlname==None:
        yamlname='/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools/config/int.REANALYSIS.yml'
    else:
        yamlname=args.yamlname
    utilities.log.info('INT YAM: NAME IS {}'.format(yamlname))
    config = utilities.load_config(yamlname)
    if args.inrange is not None:
        config['KRIGING']['VPARAMS']['range']=args.inrange
    if args.insill is not None:
        config['KRIGING']['VPARAMS']['sill']=args.insill
    utilities.log.info('Current internal yml configuration dict containing {}'.format(config))

    iometadata=args.iometadata
    inerrorfile = args.errorfile

    if args.daily:
        iometadata =  parseDateFilename(inerrorfile) # This will be used to update all output files
    else:
        iometadata =  parseWeeklyDateFilename(inerrorfile) # This will be used to update all output files

    # Fetch clamping nodes to act as boundary for kriging
    # clampfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ClampList'])
    # If clampfile not specified then try to get from the int.yml file
    clampfile = args.clampfile

    if clampfile==None:
    #    clampfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ClampList'])
        utilities.log.info('No clampfile specified. Grab the one specified in the yaml')
        clampingfile = os.path.join(os.path.dirname(__file__), "../ADCIRCSupportTools/config", self.config['DEFAULT']['ClampList'])
    #    clampfile='../ADCIRCSupportTools/config/clamp_list_hsofs_nobox.dat'

    # Fetch clamping nodes to act as boundary for kriging
    # controlfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ControlList'])
    # If controlfile not specified then try to get from the int.yml file
    controlfile = args.controlfile

    if controlfile==None:
    #    controlfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ControlList'])
        utilities.log.info('No controlfile specified. Grab the one specified in the yaml')
        controlfile = os.path.join(os.path.dirname(__file__), "../ADCIRCSupportTools/config", self.config['DEFAULT']['ControlList'])
    #    controlfile='../ADCIRCSupportTools/config/control_list_hsofs.dat '


    do_adcird_grid=True
    ###if clampfile!='Skip':
    ###    do_adcird_grid=True

    #gridjsonfile='./ADCIRCSupportTools/get_adcirc/ADCIRC/adc_coord.json'
    # gridjson is not really needed if you only want the 2D approximate image

    gridjsonfile = args.gridjsonfile
    utilities.log.info('Input files specified Errordata {}, clamp data {}, control data {}, grid json {}'.format(inerrorfile, clampfile, controlfile, gridjsonfile))

    # Fetch Lon/lat coordinates for final interpolation field. "points" format
    # SO number of LONS == number of LATS and interpolation is done based on the pairs (lon_i,lat_i) points 
    # Get desired ADCIRC coordinates for the model interpolation
    
    if (gridjsonfile!=None) and (gridjsonfile!='Skip'):
        utilities.log.info('Fetching JSON data')
        adc_json = utilities.read_json_file(gridjsonfile)
        adcgridx = adc_json['lon']
        adcgridy = adc_json['lat']

    # Start the interpolation kriging
    utilities.log.info('Begin Interpolation')
    adddata=None
    if args.inrange is not None or args.insill is not None:
        krig_object = interpolateScalerField(datafile=inerrorfile, inputcfg=config,  clampingfile=clampfile, controlfile=controlfile, metadata=iometadata, rootdir=rootdir)
    else:
        krig_object = interpolateScalerField(datafile=inerrorfile, yamlname=yamlname, clampingfile=clampfile, controlfile=controlfile, metadata=iometadata, rootdir=rootdir)

    if cv_testing:
        if classdataFile is not None:
            utilities.log.info('Attempt a CV test using {}'.format(classdataFile))
            classdata=fetch_data_metadata(classdataFile)
            print('class data {}'.format(classdata))
        utilities.log.info('Testing interpolation model using CV procedure')
        full_scores, best_score = krig_object.test_interpolationFit()
        print('Optimize {}'.format(full_scores))
        utilities.log.info('Interpolation overall best score is {}'.format(best_score))
        fullScoreDict = full_scores # {'best_score':best_score,'params':param_dict,'vparams':vparams}
        jsonfilename = 'fullCVScores.json'
        utilities.log.info('Partial CV score {}'.format(fullScoreDict))
        #print('Partial CV score {}'.format(fullScoreDict))
        cvfilename=utilities.writeDictToJson(fullScoreDict,rootdir=rootdir,subdir='interpolated',fileroot='interpolationSummaryCV',iometadata=iometadata)
        utilities.log.info('Wrote Daily CV values to {}'.format(cvfilename))

    #status = krig_object.singleStepInterpolationFit(X,Y,V,filename = 'interpolate_linear_model'+extraFilebit+iometadata+'.h5')
    #kf_dict = krig_object.test_interpolationFit(filename = 'interpolate_linear_model'+extraFilebit+iometadata+'.h5')
    #print('Optimize {}'.format(kf_dict))

    # Extra code to capture the errorsw+clamps+knn process controls for research on the surface structure
    utilities.log.info('Special error,clamp,control node assembly for interpolartion research')
    #print(krig_object.clamps)
    #print(krig_object.controls)
    #print(krig_object.data)
    temporaryNodeDict = dict()
    temporaryNodeDict['ERRORS']=krig_object.data
    temporaryNodeDict['WATER_CLAMPS']=krig_object.clamps
    temporaryNodeDict['LAND_CONTROLS']=krig_object.controls
    utilities.writeDictToJson(temporaryNodeDict,rootdir=rootdir,subdir='interpolated',fileroot='controlNodeSummary',iometadata=iometadata)

    utilities.log.info('Start interpolation')
    combineddata = np.concatenate([krig_object.data, krig_object.clamps, krig_object.controls], axis=0).astype(float)
    X,Y,V = combineddata[:,0], combineddata[:,1], combineddata[:,2]
    status = krig_object.singleStepInterpolationFit(X,Y,V,filename = 'interpolate_linear_model'+iometadata+'.h5')
    
    #############################################################################
    # Start predictions

    # Pull out the krid predicted values at the stations
    #station_gridx,station_gridy =krig_object.fetchRawInputData() # lons and lats
    df_interpolate_stations = pd.read_csv(inerrorfile, index_col=0, header=0).dropna(axis=0)

    lons=df_interpolate_stations['lon'].to_list()
    lats=df_interpolate_stations['lat'].to_list()
    meandata = df_interpolate_stations['mean'].to_list()
    utilities.log.info('Station points:Number of lons {} number of lats {}'.format(len(lons), len(lats)))

    Allvalues = krig_object.interpolationTransform(lons, lats, style='points', filename='interpolate_linear_model'+iometadata+'.h5')

    df_interpolate_stations['interpolate']=Allvalues['value'].to_list()
    krigfilename=utilities.writeCsv(df_interpolate_stations,rootdir=rootdir,subdir='interpolated',fileroot='stationSummaryKrig',iometadata=iometadata)
    utilities.log.info('Wrote Station krig values to {}'.format(krigfilename))

    # Second: test on a simple 2D grid for generating visualization work
    # Write this data to disk using a PKL simple format (lon,lat,val, Fortran order)

    gridx, gridy = krig_object.input_grid() # Grab from the config file

    # Set up a gridded plotter. THis is not working we need to pass regular gridx,gridy to the plotter.
    #g = np.meshgrid(gridx,gridy)
    #positions = np.vstack(map(np.ravel, g))
    #gridx,gridy = positions[0], positions[1] 
    df_grid = krig_object.interpolationTransform(gridx, gridy,style='grid',filename = 'interpolate_linear_model'+iometadata+'.h5')
    # Pass dataframe for the plotter
    gridz = df_grid['value'].values
    n=gridx.shape[0]
    gridz = gridz.reshape(-1, n)

    krig_object.plot_model(gridx, gridy, gridz, keepfile=True, filename='image'+iometadata+'.png', metadata=iometadata)
    krig_interfilename = krig_object.writeTransformedDataToDisk(df_grid)

    # Third: now using the real adcirc data adcirc_gridx, and adcirc_gridy 
    # NOTE these are styled as "points" not "grid"
    # Write this data to disk using an ADCIRC format. lon,lat,val F order

    if do_adcird_grid:
        adcirc_gridx=adcgridx
        adcirc_gridy=adcgridy
        utilities.log.info('Number of lons {} number of lats {}'.format(len(adcirc_gridx), len(adcirc_gridy)))
        df_adcirc_grid = krig_object.interpolationTransform(adcirc_gridx, adcirc_gridy, style='points', filename='interpolate_linear_model'+iometadata+'.h5')
        krig_adcircfilename = krig_object.writeADCIRCFormattedTransformedDataToDisk(df_adcirc_grid)
        utilities.log.info('Transformed interpolated data are in '+krig_interfilename)
        utilities.log.info('Transformed interpolated ADCIRC formatteddata are in '+krig_adcircfilename)

    #########################################################################
    # Perform some possible diagnostics
    # Get error statistics for both the visualization and actual ADC IRC nodes data
    # Generally these will not be used but are kept for developer's convenience

    # Fetch non clamped input (station-level) data for scatter plotting of stations
    newx, newy, newz = krig_object.fetchRawInputData()  # Returns the unclamped input data for use by the scatter method
    pltScatter=False
    if pltScatter:
        krig_object.plot_scatter_discrete(newx,newy,newz,showfile=True, keepfile=True, filename='image_Discrete'+iometadata+'.png', metadata='testmetadataDiscrete')

    # Characterize the errors  of the kriging method. Four subplots are automaticvally generated

    import seaborn as sns

    vizHistograms=False
    if vizHistograms:
        vmax = config['GRAPHICS']['VMAX']
        vmin = config['GRAPHICS']['VMIN']
        # Fetch the data sets for comparison. Looking to determine if error field distribution substantially changes
        # NOTE some stations have Nans retained in the final dataset (by choice) 
        df_station = pd.read_csv(inerrorfile).dropna(axis=0)
        df_2ddata = pd.read_pickle(krig_interfilename)
        df_adcircdata = pd.read_csv(krig_adcircfilename, skiprows=3)
        df_adcircdata.columns = ('node','value')

        datalist = [df_station['mean'], df_station['std'], df_2ddata['value'], df_adcircdata['value']]
        metaDataLabels = ['Mean Station errors','SD Station errors', '2d grid interpolation: error field', 'ADCIRC grid interpolation: error field']

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(0,4):
            df_Data = datalist[i] 
            inputMetadata = metaDataLabels[i]
            genSinglePlot(i+1, fig, df_Data,vmin,vmax, inputMetadata)
        fig.suptitle('TEST')
        plt.show()

    ##########################################################################
    # Test diagnostics
    # Station level plots

    visualiseErrorField=False
    # Plot the 2D fiueld with clamping nodes included
    if visualiseErrorField:
        print('plot new diagnostics')
        #selfX,selfY,selfZ = krig_object.fetchRawInputData() # Returns the unclamped input data for use by the scatter method

        selfX, selfY, selfZ = krig_object.fetchInputAndClamp()  # Returns the clamped input data for use by the scatter method

        diag.plot_interpolation_model(gridx, gridy, gridz, selfX, selfY, selfZ, metadata=iometadata)

    #vizScatterPlot=False
    #if vizScatterPlot:
    #    print('final scatter plot')
    #    diag.plot_scatter_discrete(newx, newy, newz, metadata=iometadata)

    ############################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('--errorfile', action='store', dest='errorfile',default=None, help='FQFN to stationSummaryAves_*.csv', type=str)
    parser.add_argument('--clampfile', action='store', dest='clampfile',default=None, help='FQFN to clamp_list_hsofs_nobox.dat', type=str)
    parser.add_argument('--controlfile', action='store', dest='controlfile',default=None, help='FQFN to control_list_hsofs.dat', type=str)
    parser.add_argument('--gridjsonfile', action='store', dest='gridjsonfile',default=None, help='FQFN to ADCIRC lon,lat values. if = Skip then skip ADCIRC interpolation', type=str)
    parser.add_argument('--subdir_name', action='store', dest='subdir_name', default='.',
                        help='Names highlevel $RUNTIMEDIR/subdir_name')
    parser.add_argument('--iometadata', action='store', dest='iometadata', default='',
                        help='Amends all output filenames with iometadata')
    parser.add_argument('--cv_testing', action='store_true', dest='cv_testing',
                        help='Boolean: Invoke a CV procedure prior to fitting kriging model')
    parser.add_argument('--yamlname', action='store', dest='yamlname', default=None)
    parser.add_argument('--error_histograms', action='store_true',dest='error_histograms',
                        help='Boolean: Display histograms station only, vis grid errors, and adcirc nodes')
    parser.add_argument('--outroot', action='store', dest='outroot', default=None,
                        help='Available high level output dir directory')
    parser.add_argument('--daily', action='store_true', dest='daily',
                        help='Boolean: Choose the DAILY filename nomenclature')
    parser.add_argument('--inrange', action='store', dest='inrange',default=None, help='If specified then an internal config is constructed', type=int)
    parser.add_argument('--insill', action='store', dest='insill',default=None, help='If specified then an internal config is constructed', type=float)
    parser.add_argument('--classdataFile', action='store', dest='classdataFile',default=None, help='FQFN to station metadata file.' , type=str)
    args = parser.parse_args()
    sys.exit(main(args))
