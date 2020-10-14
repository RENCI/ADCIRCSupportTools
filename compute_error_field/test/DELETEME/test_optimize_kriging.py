#
# Simple tests to demonstrate using a prototype kriging class
# Simple test to demonstrate optimization of the kriging procedure
# Specifically optimization of the vparams
#

import os,sys
import numpy as np
import pandas as pd
import compute_error_field.computeErrorField as isf
import matplotlib.pyplot as plt
import get_obs_stations.GetObsStations as rPL
import compute_error_field.interpolateScalerField as isf
from utilities.utilities import utilities as utilities
import matplotlib
import pykrige.kriging_tools as kt
from SRC.get_adcirc import Adcirc, get_water_levels63

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params

def mergeDataAndClamp(errorfile, clamp_file):
    data = pd.read_csv(clamp_file,header=0).values
    zeror = pd.read_csv(clamp_file,header=0).values
    x = np.concatenate([data,zeros],axis=0)
    return x,data,zeros

############################################################

def fetchBasedir(inconfig):
    try:
        rundir = os.environ[inconfig.replace('$','')]# Yaml call to be subsequently removed
    except:
        print('Chosen basedir invalid: '+str(inconfig))
        print('reset to CWD')
        rundir = os.getcwd()
    return rundir

############################################################
# Specify a basic rootdir for storing/fetching files

pipelineCycle = 'TestKrig'
config = utilities.load_config()
#rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')
rootdir = '/home/jtilson/ADCIRCDataAssimilation/test/TESTKRIGING'
print('rootdir is '+rootdir)
rootdir = '/home/jtilson/ADCIRCDataAssimilation/test/TESTKRIGING'
#inerrorfile='/home/jtilson/ADCIRCDataAssimilation/test/kriging_error_test_noclamp.dat'
#clampingfile='/home/jtilson/ADCIRCDataAssimilation/config/clamp_list.dat'

inerrorfile='/home/jtilson/ADCIRCDataAssimilation/test/TESTKRIGING/StationTest/kriging_error_test_noclamp.dat'
clampingfile='/home/jtilson/ADCIRCDataAssimilation/test/TESTKRIGING/StationTest/clamp_list.dat'

# Get clamping file
print(clampingfile)

print('CV interpolation procedure')
krig_object = isf.interpolateScalerField(datafile=inerrorfile, clampingfile=clampingfile,rootdir=rootdir)
gridx, gridy = krig_object.generic_grid()    # For interpolation
datax,datay,dataz = krig_object.fetchInputAndClamp() # Input data
newx,newy,newz = krig_object.fetchRawInputData() # input data clamp

param_dict = {"method": ["ordinary", "universal"],
              "variogram_model": ["linear", "power", "gaussian", "spherical"],
              "nlags": [2, 3, 4, 5, 6, 7, 8],
               "weight": [True, False]
              }

vparams = {'sill': 2, 'range': 2, 'nugget': .1}

#estimator = PyKrigeGridSearchCV(Krige(verbose=False), param_dict, verbose=False, iid=True,
#                         return_train_score=True, cv=3)

estimator = PyKrigeGridSearchCV(UniversalKriging(datax,datay,dataz,verbose=False), param_dict, verbose=False, iid=True,
                         return_train_score=True, cv=3)

data = np.concatenate((datax.reshape(-1, 1), datay.reshape(-1, 1)), axis=1)
estimator.fit(X=data, y=dataz)

print('best_score R2={:.3f}'.format(estimator.best_score_))
print('best_params = ', estimator.best_params_)

method = estimator.best_params_['method']
nlags = estimator.best_params_['nlags']
variogram_model = estimator.best_params_['variogram_model']
weight = estimator.best_params_['weight']

if method == 'ordinary':
    utilities.log.info('Ordinary kriging method selected')
    model = OrdinaryKriging(datax, datay, dataz, variogram_model=variogram_model, nlags=nlags,
                         weight=weight, variogram_parameters=vparams, verbose=False, enable_plotting=False)
else:
    utilities.log.info('Universal kriging method selected: is data not stationary ?')
    model = UniversalKriging(datax, datay, dataz, variogram_model=variogram_model, nlags=nlags,
                                  weight=weight, verbose=False, enable_plotting=False)

# Apply final interpolation onto the gridx,gridy data

df = krig_object.krigingTransform(gridx, gridy, filename = 'interpolate_model.h5' )
gridz = df['value'].values
n = gridx.shape[0]
gridz = gridz.reshape(-1,n)

## Check out some plots

# Input station data
krig_object.plot_scatter_discrete(newx,newy,newz,showfile=True, keepfile=False, filename='image_Discrete.png', metadata='testmetadataDiscrete')

# Plot interpolated data set 
krig_object.plot_model(gridx, gridy, gridz, showfile=True, keepfile=False, filename='image.png', metadata='testmetadata')

############################################################################################################################
## Wrap the kriging into a function so we can manually gridsearch the vparams list. Krige doesn't provide that kind of check
## Combine the datax and datay one and for all

data = np.concatenate((self.X.reshape(-1, 1), self.Y.reshape(-1, 1)), axis=1)
#vparams = {'sill': 2, 'range': 2, 'nugget': .1}
vparams = [ {"sill": [1,2,3,4]}, {"range": [1,2,3,4]}]
param_dict = {"method": ["ordinary", "universal"],
              "variogram_model": ["linear", "power", "gaussian", "spherical"],
              "nlags": [2, 3, 4, 5, 6, 7, 8],
               "weight": [True, False]
              }


estimator = GridSearchCV(Krige(verbose=False), vparams, verbose=False, iid=True,
                                 return_train_score=True, cv=3)





grd_search_cv = GridSearchCV(SGDClassifier(verbose=True, max_iter=1000, tol=0.0001,
    penalty='elasticnet', learning_rate='optimal', validation_fraction=0.2, random_state=42), param_grid, verbose=True, cv=10)



 # This one sort of works for getting at the inner loop
PyKrigeGridSearchCV(Krige(verbose=False), param_dict, verbose=False, iid=True,return_train_score=True, cv=3).fit(X=data, y=dataz)


#############################################################################################################################
# Outer CV level




# Low level CV first

def LowlevelCV( vparams ):

    estimator = PyKrigeGridSearchCV(Krige(verbose=False), param_dict, verbose=False, iid=True,
                         return_train_score=True, variogram_parameters=vparams, cv=3)
    data = np.concatenate((datax.reshape(-1, 1), datay.reshape(-1, 1)), axis=1)
    return (estimator.fit(X=data, y=dataz)

## Fetch data form the PyKrige CV  run 

>>> estimator.cv_results_.test_score
>>> estimator.cv_results_.keys()
dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_method', 'param_nlags', 'param_variogram_model', 'param_weight', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score', 'split0_train_score', 'split1_train_score', 'split2_train_score', 'mean_train_score', 'std_train_score'])
estimator.cv_results_['mean_test_score']


data = np.concatenate((datax.reshape(-1, 1), datay.reshape(-1, 1)), axis=1)

vparams = {'sill': 2, 'range': 2, 'nugget': .1}

## So each model has a different set of variogram_parameters
## These are NOT recommend values

# gaussian: {'sill': 2, 'range': 2, 'nugget': .1}
# linear: {'slope': 2, 'nugget': .1}
# spherical: {'sill': 2, 'range': 2, 'nugget': .1}
# exponential: {'sill': 2, 'range': 2, 'nugget': .1}
# power: {'scale': 2, 'exponent': 2, 'nugget': .1} # 0 <= exponent <=2
# ------ hole-Effect Model not recommended

#######################################################################################
## Custom model specification This works fine
# So we should optimize model first using system defaults them vparams on the best model
# May need trailing underbars

class CVKrigeParams:
    """CV optimize the indicated params
    the caller routine will genrate the best vparams
    """
    param_dict = {"method": ["ordinary", "universal"],
        "variogram_model": ["linear", "power", "gaussian", "spherical"],
        "nlags": [2, 3, 4, 5, 6, 7, 8],
        "weight": [True, False]
        }
    def __init__(self):
        print('init')
        self.estimator = PyKrigeGridSearchCV(Krige(verbose=True), 
                         param_dict, verbose=False, iid=True, return_train_score=True, cv=3)
    def fit(self, data, dataz):
        print('fit')
        self.estimator.fit(data, dataz) 
        return self.estimator.get_params()
    def score(self):
        return self.estimator.cv_results_
    def test_score(self):
        return self.estimator.cv_results_['mean_test_score']
    def train_score(self):
        return self.estimator.cv_results_['mean_train_score']
    def fit_time(self):
        return self.estimator.cv_results_['mean_fit_time']
    def score_time(self):
        return self.estimator.cv_results_['mean_score_time']
    def estimator(self):
        return estimator
    def best_params(self):
        method = estimator.best_params_['method']
        nlags = estimator.best_params_['nlags']
        variogram_model = estimator.best_params_['variogram_model']
        weight = estimator.best_params_['weight']
        return method, nlags, variogram_model, weight
    def get_params(self):
        return estimator.get_params()

#########################################################################################
## Now optimize the vparams for the custom model

grd_search_cv = GridSearchCV(CVKrigeParams(), param_dict, scoring='accuracy', verbose=True, cv=3)

grd_search_cv.fit(X=data, y=dataz)






