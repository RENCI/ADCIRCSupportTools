# Invoke the interpolationm pipeline and create ADCIRC formatted files

# In this new refactor, we always run the SingleKrigFit
# regardless of cvKrige or not. This version supports the changed
# CV procedure whicvh brute-force tests permutations of vparams
###########################################################################

## Added code to capture the set of scores for subsequenct analysis

# Need large memory to run this job
import os
import sys
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import json
import datetime
from utilities.utilities import utilities
from compute_error_field.interpolateScalerField import interpolateScalerField
import visualization.diagnostics as diag

###########################################################################
# For 2x2 histrogram plots
def genSinglePlot(i, fig, df_data,vmin,vmax,inputMetadata):
    ax = fig.add_subplot(2, 2, i)
    sns.distplot(df_data)
#    sns.scatterplot(
#        x='tsne-2d-one', y='tsne-2d-two',
#        hue=effect, # 
#        palette=sns.color_palette("hls", numeffect),
#        data=df_plotData,
#        legend=False,
#        alpha=0.4
#        )
    ax.set_title(inputMetadata, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('tight')

def main(args):
    utilities.log.info(args)

    cv_kriging = args.cv_kriging

    ###########################################################################
    # Interpolate adcirc node data using the previously generated error matrix
    # Get clamping data and prepare to merge with error file
    # Remove outliers within the compute error codes before coming here

    config = utilities.load_config()
    extraExpDir='TestInterpolate'
    #extraExpDir=''
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'],basedirExtra=extraExpDir)
   
    #Fetch Error vector ti perform kriging
    #finalf = args.errorfile
    finalf = '/home/jtilson/ADCIRCSupportTools/ADCIRC/stationSummaryAves_manualtest.csv' 

    iometadata='_TEST'
    inerrorfile = finalf

    # Fetch clamping nodes to act as boundary for kriging
    #clampfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ClampList'])
    clampfile='/home/jtilson/ADCIRCSupportTools/config/clamp_list_hsofs.dat'
    yamlname=os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'int.yml')

    # Fetch Lon/lat coordinates for final interpolation field. "points" format
    # Get desired ADCIRC coordinates for the model interpolation
    gridjsonfile='/home/jtilson/ADCIRCSupportTools/get_adcirc/ADCIRC/adc_coord.json'
 
    adc_json = utilities.read_json_file(gridjsonfile)
    adcgridx = adc_json['lon']
    adcgridy = adc_json['lat']

    # Start the kriging
    utilities.log.info('Begin Kriging')
    krig_object = interpolateScalerField(datafile=inerrorfile, yamlname=yamlname, clampingfile=clampfile, metadata=iometadata, rootdir=rootdir)

    if cv_kriging:
        extraFilebit='_CV_'
        param_dict_list = config['CVKRIGING']['PARAMS']
        vparams_dict_list = config['CVKRIGING']['VPARAMS']
        randkey = random.choice(list(config['CVKRIGING']['PARAMS'].keys()))
        randval = config['CVKRIGING']['PARAMS'][randkey]
        if not isinstance(randval, list):
            utilities.log.error("If performing CVKriging then ALL entries in the main yaml CVKRIGING must have values of type list")
    else:
        extraFilebit=''
        param_dict = config['KRIGING']['PARAMS']
        vparams = config['KRIGING']['VPARAMS']
        randkey = random.choice(list(config['KRIGING']['PARAMS'].keys()))
        randval = config['KRIGING']['PARAMS'][randkey]
        if isinstance(randval, list):
            utilities.log.error("If performing Single point Kriging then NO entries in the main yaml KRIGING can have values of type list")

    if cv_kriging:
        utilities.log.info('Building kriging model using CV procedure')
        param_dict, vparams, best_score, full_scores = krig_object.optimize_kriging(krig_object , param_dict_list, vparams_dict_list)
        utilities.log.info('Kriging best score is {}'.format(best_score))
        print('List of all scores {}'.format(full_scores))
        fullScoreDict = {'best_score':best_score,'scores': full_scores, 'params':param_dict,'vparams':vparams}
        ##jsonfilename = '_'.join(['','fullScores.json']) 
        jsonfilename = 'fullCVScores.json'
        with open(jsonfilename, 'w') as fp:
            json.dump(fullScoreDict, fp)
        
    utilities.log.info('doing a single krige using current best parameters')
    utilities.log.info('param_dict: {}'.format(param_dict))
    utilities.log.info('vparams: {}'.format(vparams))

    # Always do this final model is saved for subseqent reuse.
    status = krig_object.singleStepKrigingFit( param_dict, vparams, filename = 'interpolate_model'+extraFilebit+iometadata+'.h5')

    #############################################################################
    # Start predictions

    # First: test on a simple 2D grid for generating visualization work
    # Write this data to disk using a PKL simple format (lon,lat,val, Fortran order)
    gridx, gridy = krig_object.input_grid() # Grab from the config file
    df_grid = krig_object.krigingTransform(gridx, gridy,style='grid',filename = 'interpolate_model'+extraFilebit+iometadata+'.h5')
    # Pass dataframe for the plotter
    gridz = df_grid['value'].values
    n=gridx.shape[0]
    gridz = gridz.reshape(-1, n)
    krig_object.plot_model(gridx, gridy, gridz, keepfile=True, filename='image'+iometadata+'.png', metadata=iometadata)
    krig_interfilename = krig_object.writeTransformedDataToDisk(df_grid)

    # Second: now using the real adcirc data adcirc_gridx, and adcirc_gridy 
    # NOTE these are styled as "points" not "grid"
    # Write this data to disk using an ADCIRC format. lon,lat,val F order
    adcirc_gridx=adcgridx
    adcirc_gridy=adcgridy
    df_adcirc_grid = krig_object.krigingTransform(adcirc_gridx, adcirc_gridy, style='points', filename='interpolate_model'+extraFilebit+iometadata+'.h5')
    krig_adcircfilename = krig_object.writeADCIRCFormattedTransformedDataToDisk(df_adcirc_grid)
    #krig_interfilename = krig_object.writeTransformedDataToDisk(df_grid)

    utilities.log.info('Transformed interpolated data are in '+krig_interfilename)
    utilities.log.info('Transformed interpolated ADCIRC formatteddata are in '+krig_adcircfilename)

    #########################################################################
    # Perform some possible diagnostics
    # Get error statistics for both the visualization and actual ADC IRC nodes data
    # Generally these will not be used but are kept for developer's convenience

    # Fetch non clamped input (station-level) data for scatter plotting of stations
    newx, newy, newz = krig_object.fetchRawInputData()  # Returns the unclamped input data for use by the scatter method
    krig_object.plot_scatter_discrete(newx,newy,newz,showfile=False, keepfile=True, filename='image_Discrete'+iometadata+'.png', metadata='testmetadataDiscrete')

    # Characterize the errors  of the kriging method. Four subplots are automaticvally generated

    import seaborn as sns

    #vizHistograms=True
    #if vizHistograms:
    vmax = config['GRAPHICS']['VMAX']
    vmin = config['GRAPHICS']['VMIN']

    # Fetch the data sets for comparison. Looking to determine if error field distribution substantially changes

    df_station = = pd.read_csv(inerrorfile)
    df_2ddata = pd.read_pickle(krig_interfilename)
    df_adcircdata = pd.read_csv(krig_adcircfilename, skiprows=3)

    datalist = [df_station['mean'], df_station['std'], df_test['value'], df_test['value']]
    metaDataLabels = ['Mean Station errors','SD Station errors', '2d grid interpolation: error field', 'ADCIRC grid interpolation: error field']


    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(0,4):
        df_Data = datalist[i] 
        inputMetadata = metaDataLabels[i]
        print(inputMetadata)
        print('start plot')
       genSinglePlot(i+1, fig, df_Data,vmin,vmax, inputMetadata)
    fig.suptitle('TEST')

 genSinglePlot(i, fig, df_data,vmin,vmax,inputMetadata)


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

    visualiseErrorField=True
    # Plot the 2D fiueld with clamping nodes included
    if visualiseErrorField:
        print('plot new diagnostics')
        #selfX,selfY,selfZ = krig_object.fetchRawInputData() # Returns the unclamped input data for use by the scatter method
        selfX, selfY, selfZ = krig_object.fetchInputAndClamp()  # Returns the clamped input data for use by the scatter method
        diag.plot_interpolation_model(gridx, gridy, gridz, selfX, selfY, selfZ, metadata=iometadata)

    #vizScatterPlot=True
    #if vizScatterPlot:
    #    print('final scatter plot')
    #    diag.plot_scatter_discrete(newx, newy, newz, metadata=iometadata)

    ############################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Names highlevel Experiment-tag value')
    parser.add_argument('--cv_kriging', action='store_true', dest='cv_kriging',
                        help='Boolean: Invoke a CV procedure prior to fitting kriging model')
    parser.add_argument('--station_missing_threshold', action='store', dest='station_missing_threshold', default=100,
                        help='Float: maximum percent missing station data')
    parser.add_argument('--error_histograms', action='store_true',dest='error_histograms',
                        help='Boolean: Display histograms station only, vis grid errors, and adcirc nodes')
    args = parser.parse_args()
    sys.exit(main(args))
