#!/usr/bin/env python

# Class interpolateScalerField(object):
# A class to manage the construction the interpolated fields of the ADCIRC-OBSERVATONS errorfields
# Kriging methods are used to to build a model of the station-errorfield.
# From this two things typically happen:
#     1) The user may want to visdually inspect the interpolation by constructing a geo-ref 
#     visualization plot. This plot is always createwd but under ordinary execution conditions
#     is written to disk.
#     2) The user will want to interpolate the ADCIRC field and write that data to a new file in 
#     a (csv) format suitable for passing to ADCIRC.
#
# As a convenience, to construct the visdualization grid, the YAML file has been populated with 
# some typical values. Namely, 
#REGRID: &regrid
#  InterpToAdcirc: true
#  RECT:
#    lowerleft_x: -100
#    lowerleft_y: 20
#    res: .1  # resolution in deg
#    nx: 400
#    ny: 300
#
# The kriging procedure is two step:
#    1) For fitting the data the model is compouted and then saved into the file 
#    rootdir/models/interpolate_model_metadata.h5
#    2) Subsequent interpolations may be performed by reading this model file.
#
# Two kinds of Kriging procedures are included. 
#    1) The recommended method is a simple kriging call (krig_object.singleStepKrigingFit) which will 
#    generate the model. It uses UniversalKriging and a gaussian variogram model with variogram parameters of
#    vparams = {'sill': 2, 'range': 2, 'nugget': .05}. Some cursory tests on calm East Coast 
#    conditions suggests the imputed error field is verey insensitive to the specific vparams.
#
#    2) A user can also perform a CrossValidation procedure. However, because of limitationsa in 
#    the PyKrige methods, one can not optimize vparams. The CV procedure joins with a simple gridsearch as:
#     param_dict = {"method": ["ordinary", "universal"],
#                   "variogram_model": ["linear", "power", "gaussian", "spherical"],
#                   "nlags": [2, 4, 6, 8],
#                   "weight": [True, False]
#                   }
# Once the best parameters are found, the model is constructed and stored to disk.
#
# The errorfield data is a CSV dataframe with the format:
#     stationid,lon,lat,Node,mean,std
# The clampingdata are long,lat Zeros points. These are intended to 
# drive the interpolated field to zero beyond a distance of the stations. Its format is:
#     lon, lat, val (=0.0)
# The clamping file generally doesn't change for a particular geographic region so for 
# convenience the user simply specified it in the YAML file. The default settings are:
#     clampfile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['ClampList']) 
# 
#############################################################

import sys, os
import random
import pandas as pd
import numpy as np
import pykrige.kriging_tools as kt
from compute_error_field.KrigeReimplemented import Krige as newKrige
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.rk import Krige
from pykrige.compat import GridSearchCV
from sympy import pretty_print as pp, latex
from sklearn.externals import joblib
#from sklearn.externals.joblib import Parallel, delayed
from utilities.utilities import utilities
import datetime

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import itertools
from operator import itemgetter
############################################################################
# Some basic functions

print("Numpy Version = {}".format(np.__version__))

class interpolateScalerField(object):
    """
    Class to manage interpolating the error fields. Currently only implements kriging
    Pass in the fully qualified filenames for the observations data and the adcirc data
    Perform some rudimentary checks on times/stations and then compute an error matrix
    """ 
    def __init__(self, datafile='./kriging_error_test_noclamp.dat',
                 yamlname=os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'int.yml'),
                 clampingfile=None, model='kriging', metadata='', rootdir=None):
        """
        interpolateScalerField constructor.

        Parameters:
            inerrorfile: (str) fullpath name of the CSV datafile containing the errorfield
            clampfile: (str) fullpath name of the CSV datafile containing the clampfield 
            metadata: (str) use to amend output filenames
            rootdir: (str) rootdir/interpolated location to save output files

        Results: (depending on choices by the user)
            ADCIRC formatted field: rootdir/interpolated/ADCIRC_interpolated_wl_metadata.csv
            2D Visualization field: rootdir/interpolated/interpolated_wl_metadata.pkl
            model: rootdir/models/interpolate_model_metadata.h5 (single kriging)
            model: rootdir/models/interpolate_cv_model_metadata.h5 (CV kriging)
            station error image: rootdir/images/image_Discrete_metadata.png
            2D field image: rootdir/images/image_metadata.png 
        """
        self.f = datafile # Input error field of long,lat,values including clamping zeros
        self.c = clampingfile
        self.model = model
        self.config = utilities.load_config(yamlname)
        self.iometadata = metadata
        if clampingfile == None:
            utilities.log.info('No input clampfile provided. Try to fetchj from config yml {}'.format(yamlname))
            clampfile = os.path.join(os.path.dirname(__file__), "../config", self.config['DEFAULT']['ClampList'])
        if datafile != None and clampingfile != None:
            self.X, self.Y, self.Values, self.inX, self.inY, self.inV = self.readAndProcessData( self.f, self.c )
        else:
            utilities.log.info('interpolateScalerField initialized with no input file nor clamping file: Proceeding')
        self.rootdir = rootdir
        if self.rootdir == None:
            utilities.log.error('No rootdir was specified')
            sys.exit('No rootdir was specified')

    def fetchInputAndClamp(self):
        """
        Return the X,Y,Value data and clamped input data.

        Results:
            X: numpy.ndarray list of lons
            Y: numpy.ndarray list of lats
            Values: numpy.ndarray list of krigevalues
        """
        return self.X, self.Y, self.Values

    def fetchRawInputData(self):
        """
        Return the X,Y,Value data for the unclamped input data

        Results:
            inX: numpy.ndarray list of lons
            inY: numpy.ndarray list of lats
            inV: numpy.ndarray list of krigevalues
        """
        return self.inX, self.inY, self.inV

    def generic_grid(self):
        """
        Build 2D grid for interpolating the kriging data on a visualization.
        grid. This is not used anymore.Superceded by using YAML file.

        Results:
            x: numpy.ndarray of lons
            y: numpy.ndarray of lats
        """
        print('Using a default interpolation grid that is intended only for testing')
        lowerleft_x = -90
        lowerleft_y = 20
        res = .05  # resolution in deg
        nx = 400
        ny = 500
        x = np.arange(lowerleft_x, lowerleft_x + nx * res, res)
        y = np.arange(lowerleft_y, lowerleft_y + ny * res, res)
        return x, y

    def input_grid(self):
        """
        Build 2D grid for interpolating the kriging data on a visualization
        grid.

        Parameters:
            read data from ther main YAML file

        Results:
            x: numpy.ndarray of lons
            y: numpy.ndarray of lats
        """
        print('input interpolation grid that is intended only for testing')
        config = self.config # We have previously only kept REGRID
        lowerleft_x = config['REGRID']['RECT']['lowerleft_x']
        lowerleft_y = config['REGRID']['RECT']['lowerleft_y']
        res = config['REGRID']['RECT']['res']
        nx = config['REGRID']['RECT']['nx']
        ny = config['REGRID']['RECT']['ny']
        x = np.arange(lowerleft_x, lowerleft_x + nx * res, res)
        y = np.arange(lowerleft_y, lowerleft_y + ny * res, res)
        return x, y

    def readAndProcessData(self, f, c):
        """
        Read the input data file (f) and the input clamping file (c).
        build x,y,z data that integrates the data for interpolation.
        Retain fx,fy,fz data for other visualizastion purposes.
        The f+c data get randomly shuffled because if you choose a CV procedure
        it is likely to get a cv fold of only zero values.
        nans in either f or c will cause an abort.

        Parameters:
            f: fullpath filename to station errorfile data
            c: fullpath filename to station clamping data

        Results:
            Xpoints: numpy.ndarray of lons (integrated with clamp)
            Ypoints: numpy.ndarray of lats (integrated with clamp)
            Zpoints: numpy.ndarray of values (integrated with clamp)
            inX: numpy.ndarray of lons (no clamp)
            inY: numpy.ndarray of lats (no clamp)
            inZ: numpy.ndarray of vals (no clamp)
        """
        indataAll = pd.read_csv(f, header=0)
        indataAll.dropna(axis=0, inplace=True) # Need this incase we save summaries with nana
        indata = indataAll[['lon', 'lat', 'mean']].values
        zeros = pd.read_csv(c,header=0).values  # Note we have a header here
        data = np.concatenate([indata, zeros], axis=0).astype(float)
        utilities.log.debug(indata)
        # print(data)
        # we want headers to be included
        np.random.shuffle(data) # incase we want to do CV studies
        Xpoints, Ypoints, Valuepoints = data[:,0], data[:, 1], data[:, 2]
        inX, inY, inV = indata[:,0], indata[:, 1], indata[:, 2]
        if not np.isnan(Xpoints).any() and not np.isnan(Ypoints).any() and not np.isnan(Valuepoints).any():
            return Xpoints, Ypoints, Valuepoints, inX, inY, inV
        else:
            utilities.log.error('Some of the input data are nans: Aborting ')
            sys.exit('Some of the input data are nans: Aborting ')
        return Xpoints, Ypoints, Valuepoints, inX, inY, inV

##
## Modify this to accept param,vprams and a filename on input
##
    #def singleStepKrigingFit(self, param_dict={'method':'ordinary','variogram_model':'gaussian'}, vparams={'sill':2,'range':2,'nugget':.05},filename = 'interpolate_model.h5'):
    def singleStepKrigingFit(self, param_dict=None, vparams=None,filename = 'interpolate_model.h5'):
        """
        Build a kriging model Universla kriging, variogram_model='gaussian'
        n_lag=6, weight=False, 
        vparams = {'sill': 2, 'range': 2, 'nugget': .05}.

        Parameters:
            filename: (str) name to save model

        Results:
            newfilename: rootdir/models/interpolate_model_metadata.h5
        """
        if (param_dict==None) or (vparams==None):
            utilities.log.info('No params_dict or vpamrs passed to singleStep. Try to read from config')
            param_dict = self.config['KRIGING']['PARAMS']
            vparams = self.config['KRIGING']['VPARAMS']
            randkey = random.choice(list(self.config['KRIGING']['PARAMS'].keys()))
            randval = self.config['KRIGING']['PARAMS'][randkey]
            print('Single Krig param get {}'.format(vparams))
            if isinstance(randval, list):
                utilities.log.error("If performing Single point Kriging then NO entries in the main yaml KRIGING can have values of type list")
        subdir = "models"
        status = False
        method = param_dict['method'] # Must have at least this
        param_dict.pop('method')
        utilities.log.info('single Krig: vparams {}'.format(vparams))
        utilities.log.info('single Krig: params_dict {}'.format(param_dict))
        if method == 'ordinary':
            utilities.log.info('Ordinary kriging method selected')
            model = OrdinaryKriging(self.X, self.Y, self.Values, **param_dict,
                                 variogram_parameters=vparams, verbose=False, enable_plotting=False)
        else:
            utilities.log.info('Universal kriging method selected: is data not stationary ?')
            model = UniversalKriging(self.X, self.Y, self.Values, **param_dict,
                                  variogram_parameters=vparams, verbose=False, enable_plotting=False)
        imgdir = self.rootdir # fetchBasedir(self.config['DEFAULT']['RDIR'].replace('$',''))# Yaml call to be subsequently removed except:
        newfilename = utilities.getSubdirectoryFileName(imgdir, subdir, filename)
        try:
            joblib.dump(model, newfilename)
            status = True
            utilities.log.info('Saved model file '+str(newfilename))
        except:
            utilities.log.error('Could not dump model file to disk '+ newfilename)
        # kt.write_asc_grid(x, y, z_krige, filename="output.asc") # No need to write to an output file at this time
        return status

    def krigingTransform(self, gridx, gridy, style = 'grid',  filename = 'interpolate_model.h5'):
        """
        Load the available model and apply to the provided gridx,gridy terms
        If style is "grid' then x,y are treated are ranges and interpolation 
        is performed on a 2D grid with x*y 
        number of points. This is intended for the 2D visualization grids.
        If style is set to 'points' then we will interpolate for ADCIRC and the x,y 
        are treated as pairs of points.

        Parameters:
            gridx: numpy.ndarray of lons.
            gridy: numpy.ndarray of lats.
            filename: (str) name of output model file.

        Results:
            df: output interpolation in linearized Fortran order
        """
        subdir = "models"
        #config = utilities.readConfigYml(self.yamlfile)
        imgdir = self.rootdir # fetchBasedir(self.config['DEFAULT']['RDIR'].replace('$',''))# Yaml call to be subsequently removed
        newfilename = utilities.getSubdirectoryFileName(imgdir, subdir, filename)
        try:
            model = joblib.load(newfilename)
        except:
            utilities.log.error('Failed to load model '+newfilename)
        utilities.log.info('Krige using a style of '+style)
        z_krige, ss = model.execute(style, gridx, gridy)
        # kt.write_asc_grid(x, y, z_krige, filename="output.asc") # No need to write to an output file at this time
        ##print('confidence interval {}'.format(ss))
        ##print('z_krige size ')
        utilities.log.info('z_krige shape is {}.'.format(str(z_krige.shape)))
        d = []
        xl = len(gridx)
        yl = len(gridy)
        if style=='grid':
            for y in range(0,yl):
                gy = gridy[y]
                for x in range(0,xl):
                    zval = z_krige[y,x]
                    d.append((gridx[x], gy, zval)) # This arrangement gave the correct test plot
        else:
            for y in range(0,yl):
                gy = gridy[y]
                gx = gridx[y]
                zval = z_krige[y]
                d.append((gx, gy, zval))
        df = pd.DataFrame(d,columns=['lon','lat','value']) #(Z is 500 by 400 in lat major order)
        return df 

    def writeTransformedDataToDisk(self, dfin):
        """
        Write data that has been transformed by the kriging model to disk as a pkl.

        Parameters:
            dfin: dataframe of format (lon,lat,value). If a 2D data set such as the 
                visualization grid the data has been linearized in Fortran order

        Results:
            dfin saved to rootdir/interpolated/interpolated_wl_metadata.pkl
        """
        interdir = self.rootdir
        self.newfilename = utilities.getSubdirectoryFileName(interdir, 'interpolated', 'interpolated_wl'+self.iometadata+'.pkl')
        dfin.to_pickle(self.newfilename)
        utilities.log.info('Wrote current interpolated grid to disk '+self.newfilename)
        return self.newfilename

    def writeADCIRCFormattedTransformedDataToDisk(self, dfin):
        """
        Write data that has been transformed by the kriging model to disk as a CVS 
        file suitable for reading by ADCIRC.

        Parameters:
            dfin: dataframe of format (lon,lat,value).

        Results:
            dfin saved to rootdir/interpolated/ADCIRC_interpolated_wl_metadata.csv
        """
        interdir = self.rootdir
        self.newfilename = utilities.getSubdirectoryFileName(interdir, 'interpolated', 'ADCIRC_interpolated_wl'+self.iometadata+'.csv')
        df_adcirc = dfin['value'].to_frame().astype(str)
        df_adcirc['node']=(df_adcirc.index+1).astype(str) # NODEID is index id +1
        d = []
        d.append('# Interpolated field')
        d.append('99999.9')
        d.append('0.0')
        for index,row in df_adcirc.iterrows():
            nd = row['node']
            nv = row['value']
            d.append(nd+','+nv)
        with open(self.newfilename, mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(d))
        utilities.log.info('Wrote current interpolated ADCIRC grid to disk')
        return self.newfilename
##
## New layer to loop over vparams and call CV kriging
##
#    def optimize_kriging(self, krig_object, param_dict, vparams_dict ):
    def optimize_kriging(self, krig_object):
        """
        bestname is the model name but we do not really use it after this
        """
        utilities.log.info('Building kriging model using new optimize_kriging procedure')
        # Refactored config code to here
        param_dict = self.config['CVKRIGING']['PARAMS']
        vparams_dict = self.config['CVKRIGING']['VPARAMS']
        randkey = random.choice(list(self.config['CVKRIGING']['PARAMS'].keys()))
        randval = self.config['CVKRIGING']['PARAMS'][randkey]
        if not isinstance(randval, list):
            utilities.log.error("If performing CVKriging then ALL entries in the main yaml CVKRIGING must have values of type list")
        # build a new vparams fort passing to the CV optimizer
        keys, values = zip(*vparams_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for v in permutations_dicts:
            utilities.log.debug('dict list {}'.format(v))
        overall = []
        # For each set of params perform a conventinal CV on the set of params
        fullScores = list()
        for vparams in permutations_dicts:
            utilities.log.info('Next iteration: vparams {}'.format(vparams))
            best_param, best_score, currentScores = krig_object.CVKrigingFit(param_dict=param_dict,vparams=vparams)
            bestdict = {'param':best_param, 'vparams': vparams,'score':best_score}
            overall.append(bestdict)
            fullScores.append(currentScores)
        flatFullScores = list(itertools.chain.from_iterable(fullScores))
        #print('flat combined scores {}'.format(flatFullScores))
        utilities.log.debug('final set of best params, vparams, and scores {}'.format(overall))
        #print(sorted(overall, key=itemgetter('score'),reverse=True))
        best = sorted(overall, key=itemgetter('score'),reverse=True)
        utilities.log.info('Final chosen best Krig model is {}'.format(best[0]))
        return best[0]['param'], best[0]['vparams'], best[0]['score'], flatFullScores
##
## Modify this to simply return the best param, vparam for a subsequent call to SingleKriging
##
    def CVKrigingFit(self, param_dict, vparams):
        """
        Build a kriging model performing a basic CV procdure: Choose the best parameters
        gridsearch optimize the param_dict. 
        while holding vparams as vparams = {'sill': 2, 'range': 2, 'nugget': .1}
    
        For statistical analysis, we return all the test scores to the calling program 

        Parameters:
            filename: (str) name to save model
            param_dict: dict of params values (not lists)
            vparams: dict of vparam values ( not lists) 

        Results:
            CV optimized best params for the input vparams
            CV best score conditioned on the input vparams
            parrot the input vparams
        """
        utilities.log.info("New kriging CV procedure")
        param_dict = param_dict
        utilities.log.info('Must check and remove any linear or power specs from the param_dict object')
        model_list = param_dict['variogram_model']
        model_list = list(v for v in model_list if v!='linear' and v!='power' )
        param_dict['variogram_model'] = model_list
        utilities.log.info('Params for current CV {}'.format(param_dict))
 
        #scoring='accuracy'
        estimator = GridSearchCV(newKrige(variogram_parameters=vparams), param_dict, error_score='raise', scoring='r2',verbose=True, iid=True,
                                 return_train_score=True, cv=10)
        data = np.concatenate((self.X.reshape(-1, 1), self.Y.reshape(-1, 1)), axis=1)
        estimator.fit(X=data, y=self.Values)
        # This doesn't help print('Print fixed vparams estimator {}'.format(estimator))
        print("\nCV results::")
        print(estimator)

        currentScores = None
        if hasattr(estimator, "cv_results_"):
            for key in [
                "mean_test_score",
                "mean_train_score",
                "param_method",
                "param_variogram_model"]:
                utilities.log.info("New key info - {} : {}".format(key, estimator.cv_results_[key]))
        # Repeat to keep original code
        if hasattr(estimator, "cv_results_"):
            key='mean_test_score'
            currentScores = estimator.cv_results_[key]
        utilities.log.debug('show current scores  for vparams {} {}'.format(vparams,currentScores))

        if hasattr(estimator, 'best_score_'):
            utilities.log.info('best_score R2={:.3f}'.format(estimator.best_score_))
            utilities.log.info('best_params = {}', estimator.best_params_)
            utilities.log.info('vparams {}'.format(('Params for current CV {}'.format(vparams))))
  
        return estimator.best_params_, estimator.best_score_, currentScores

    def plot_model(self, x, y, z_krige, filename='image.png', metadata='Kriging/Python of Matthew Error Vector',
                   keepfile=False, showfile=False):
        """
        Basic plotter to display a 2D interpolatrion field. 

        Parameters:
            x: numpy.ndarray of lons
            y: numpy.ndarray of lats
            z_krige: numpy.ndarray of values (interpolated errors)
            filename: ('image.png') image filename. metradata will get incorporated
            keepFile: (bool) (True) if True then file will be saved
            showFile: (bool) (False) if True plot will be displayed

        Results:
            rootdir/images/image_metadata.png
        """
        subdir = "images"  # The yaml imgdir/subdir for storing images
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.pcolormesh(x, y, z_krige,
                      cmap=plt.cm.jet,
                      vmin=-.2, vmax=1.15)
        ax.scatter(self.X, self.Y, s=100, marker='o',
                   c=self.Values, cmap=plt.cm.jet, edgecolor='k',
                   vmin=-.2, vmax=1.15)
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])
        ax.set_title(metadata)
        plt.draw()
        imgdir = self.rootdir # fetchBasedir(self.config['DEFAULT']['RDIR'].replace('$','')) # Yaml call to be subsequently removed
        self.plotfilename = utilities.getSubdirectoryFileName(imgdir, subdir, filename)
        if keepfile:
            try:
                utilities.log.info(self.plotfilename)
                plt.savefig(self.plotfilename, bbox_inches='tight')
                utilities.log.info('Saved plot file '+str(self.plotfilename))
            except:
                utilities.log.error('Failed to save interpolation image file ' + filename)
        if showfile:
            plt.show()
        plt.close(fig)
        return self.plotfilename


    def plot_scatter_discrete(self, x, y, values, filename='image_discrete.png',
                   metadata='Kriging/Python of Matthew Error Vector: Stations',keepfile=False, showfile=False):
        """ 
        Basic plotter to display a small number of discrete points such as values at the stations.
        Generally useful to pass the clamping points as visual cues.

        Parameters:
            x: numpy.ndarray of lons
            y: numpy.ndarray of lats
            values: numpy.ndarray of values (interpolated errors)
            filename: ('image_discrete.png') image filename. metradata will get incorporated
            keepFile: (bool) (True) if True then file will be saved
            showFile: (bool) (False) if True plot will be displayed

        Results:
            rootdir/images/image_Discrete_metadata.png
        """
        subdir = "images" # The yaml imgdir/subdir for storing images
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        cmap = plt.get_cmap('seismic')
        cmap.set_under('gray')
        cax = ax.scatter(x, y, c=values, s=100, cmap=cmap, vmin=values.min(), vmax=values.max())
        fig.colorbar(cax)
        ax.set_title(metadata)
        plt.draw()
        imgdir = self.rootdir # fetchBasedir(self.config['DEFAULT']['RDIR'].replace('$','')) # Yaml call to be subsequently removed
        self.scatterfilename = utilities.getSubdirectoryFileName(imgdir, subdir, filename)
        if keepfile:
            try:
                utilities.log.info(self.scatterfilename)
                plt.savefig(self.scatterfilename, bbox_inches='tight')
                utilities.log.info('Saved discrete plot file '+str(self.scatterfilename))
            except:
                utilities.log.error('Failed to save discrete image file ' + filename)
        if showfile:
            utilities.log.debug('Dumping plot file')
            plt.show()
        plt.close(fig)
        return self.scatterfilename


