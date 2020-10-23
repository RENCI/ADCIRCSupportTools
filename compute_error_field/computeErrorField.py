#!/usr/bin/env python

# Class to manage the construction of the ADCIRC-OBSERVATONS error fields
# Mostly this class manages filenames as computing the error is easy. What is hard
# is we wish to average error files over a user-specified number of cycles.
#
# Check the station df_final error measures. If z-score > 3 remove them 

import os, sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utilities.utilities import utilities
from scipy import stats

class computeErrorField(object):
    """ Pass in the fully qualitied filenames for the observations data and the adcirc data
    Perform some rudimentary checks on times/stations and then compute an error matrix
    """
    #def __init__(self, obsf, adcf, meta, bound_lo=None, bound_hi=None, n_cycles=4, n_pad=1, n_period = 12, n_tide = 12.42):
    def __init__(self, obsf, adcf, meta, yamlname=os.path.join(os.path.dirname(__file__), '../config', 'err.yml'),
                 bound_lo=None, bound_hi=None, rootdir=None, aveper=None, zthresh=3):
        self.obs_filename = obsf
        self.adc_filename = adcf
        self.meta_filename = meta
        self.config = utilities.load_config(yamlname)
        self.n_pad = self.config['TIME']['n_pad']
        #self.n_cycles = self.config['TIME']['AvgPer']
        self.n_cycles = self.config['TIME']['AvgPer'] if aveper==None else aveper
        self.bound_lo = bound_lo
        self.bound_hi = bound_hi
        self.n_period = self.config['TIME']['n_period']
        self.n_tide = self.config['TIME']['n_tide']
        self.zthresh = zthresh 
        utilities.log.debug("ADCF filename "+adcf)
        utilities.log.debug("OBSF filename "+obsf)
        self.rootdir = rootdir
        if self.rootdir == None:
            utilities.log.error('No rootdir was specified')
        try:
            self.df_obs_wl = pd.read_pickle(self.obs_filename)
        except FileNotFoundError:
            raise IOerror("Failed to read %s" % self.obs_filename)
            utilities.log.error("Failed to read %s" % self.obs_filename)
        try:
            self.df_adc_wl = pd.read_pickle(self.adc_filename)
        except:
            raise IOerror("Failed to read %s" % self.adc_filename)
            utilities.log.error("Failed to read %s" % self.adc_filename)
        try:
            self.df_meta = pd.read_pickle(self.meta_filename)
            self.df_meta.set_index('stationid', inplace=True)
        except:
            raise IOerror("Failed to read %s" % self.meta_filename)
            utilities.log.error("Failed to read %s" % self.meta_filename)
        utilities.log.debug('input ADC wl data are {}'.format(str(self.df_adc_wl)))
        utilities.log.debug('input OBS wl data are {}'.format(str(self.df_obs_wl)))
        print(self.df_adc_wl)
        print(self.df_obs_wl)

    def _intersectionStations(self):
        """ Reduces the columns to the greatest common number
        """
        int_stations = self.df_adc_wl.columns & self.df_obs_wl.columns
        #print('Number of intersected stations is '+str(len(int_stations)))
        utilities.log.info('Number of intersected stations is '+str(len(int_stations)))
        self.df_adc_wl = self.df_adc_wl[int_stations]
        self.df_obs_wl = self.df_obs_wl[int_stations]

    def _constructMetaData(self):
        """Build information for storage to disk
        """
        Meta = { 'OBS_filename': self.obs_filename,
                 'ADC_filename':self.adc_filename,
                 'Tide padding': self.n_pad,
                 'Ave periods': self.n_cycles,
                 'Bounds lo': self.bound_lo,
                 'Bounds hi': self.bound_hi,
                 'Time steps per period': self.n_period,
                 'Tidal period ':self.n_tide}
        return Meta

    def _intersectionTimes(self):
        """Keep only those times found in both obs and adc
        """
        inttimes = self.df_adc_wl.index & self.df_obs_wl.index
        utilities.log.info('Number of intersected times is '+str(len(inttimes)))
        self.df_adc_wl = self.df_adc_wl.loc[inttimes]
        self.df_obs_wl = self.df_obs_wl.loc[inttimes]

    def _findStationOutliers(self):
        """ Go through the Summary error data and look for stations with 
        outlier status nthe errors. This may imply outside forcing functions 
        and warrents removal of the station
        must drop all NANs to perform the analysis. DO not want to permenantly remove them
        because some users want the nan values (station list)in the final file
        """
        utilities.log.info('Requested check for station error outlier status pre-remove nans. Using a zthresh of {}.'.format(self.zthresh))
        z = np.abs(stats.zscore(self.df_final['mean'].dropna(axis=0)))>self.zthresh
        #print(stats.zscore(self.df_final['mean'].dropna(axis=0)))
        droplist = self.df_final.dropna(axis=0)[z].index.tolist()
        utilities.log.info('Number of stations drop because of outliers is {}.'.format(str(len(droplist))))
        print(str(droplist))
        return droplist 

# Maybe check that bound_lo < bound_hi ?
# NOTE: user [passing in bound now will not work becasue they would need toknow exactlt what the tidal times are
# So remove that as an option for now and just keep cycles
    def _applyTimeBounds(self):
        """Take the current input bound_lo/hi and clamp the time indexes
        For now hard fail if the bounds are wrong. A None is okay and simply grabs
        the end points for ADC
        force format to datetime64
        For now bound_up will always be the ADC highest value (=now). bound_lo will be highest-n_cycles*n_period)
        which accounts for the amount of time used to average. We can always overwrite the bounds manually
        """
#       self.bound_lo = df_adc_wl.index.min() if self.bound_lo==None else  np.datetime64(self.bound_lo)
        if len(self.df_adc_wl.index) < self.n_cycles*self.n_period:
            #print('Averaging bounds range too wide for actual data. ADC only of length '+str(len(self.df_adc_wl.index)))
            utilities.log.error('Averaging bounds range too wide for actual data. ADC only of length '+str(len(self.df_adc_wl.index)))
            sys.exit('Failed averaging bounds setting')
        self.bound_hi = self.df_adc_wl.index.max() if self.bound_hi==None else  np.datetime64(self.bound_hi)
        #self.bound_lo = self.df_adc_wl.index.min() if self.bound_lo==None else np.datetime64(self.df_adc_wl.index[-(self.n_cycles*self.n_period)])
        self.bound_lo = self.df_adc_wl.index[-(self.n_cycles*self.n_period)] if self.bound_lo==None else np.datetime64(self.bound_lo)
        print(self.df_adc_wl)
        print(self.bound_lo)
        print(self.bound_hi)
        print(self.n_cycles)
        print(self.n_period)
        utilities.log.info('bounds (inclusive) lo and hi are '+str(self.bound_lo)+' '+str(self.bound_hi))
        if self.df_adc_wl.index.max() < self.bound_hi:
            utilities.log.error('input hi bound is too high for ADC. Max is '+str(self.df_adc_wl.index.max()))
            sys.exit('Wrong upper bound')
        if self.df_adc_wl.index.min() > self.bound_lo:
            utilities.log.error('input lo bound is too low for ADC. Min is '+str(self.df_adc_wl.index.min()))
            sys.exit('Wrong lower bound')
        # NOTE we could conceivably reset here: NOTE also we must have already intersected
        # times fo rthis to work as follows
        removeRange = (self.df_adc_wl.index < self.bound_lo) | (self.df_adc_wl.index > self.bound_hi)
        removeTimes = self.df_adc_wl[ removeRange ].index
        self.df_adc_wl.drop(removeTimes,axis=0, inplace=True)
        self.df_obs_wl.drop(removeTimes,axis=0, inplace=True)
        utilities.log.info('After bounds constraint sizes are ADC '+str(self.df_adc_wl.shape)+' OBS '+str(self.df_obs_wl.shape))

    def _tidalCorrectData(self):
        """For ADC and OBS. transform data (inplace) to interpolate on a diurnal time of period n_tide = 12.42
        hours (3726 secs). Once done remove the input hourly data keeping only the newly interpolated 
        results
        Expects intersection of times and bounds to have been applied. No checks are performed
        NOTE timein, timeout are interpolation ranges and not actual data fetches. So extenbdingh timeout
        beyond the ADCIRC 'now' state is okay.
        """
        normalRange = self.df_adc_wl.index.values
        timein, timeout = normalRange[0], normalRange[-1]
        normalRange = pd.date_range(timein, timeout, freq='3600S') # NEEDED becasue of datetime format differences
        n_period = self.n_period
        n_tide = self.n_tide
        n_pad = self.n_pad
        if n_period != 12:
            sys.exit('Interpolation code currently only tested for n_period=12')
        timef =  int(3600*n_tide/n_period) # Always scale to an hour (3600s)
        diurnalRange = pd.date_range(timein, timeout+np.timedelta64(n_pad,'h'), freq=str(timef)+'S')
        utilities.log.info('interpolation timein '+str(timein))
        utilities.log.info('interpolation timeout with + hourly padding '+str(timeout+np.timedelta64(n_pad,'h')))

        df_in= self.df_adc_wl.copy()
        df_x = pd.DataFrame(diurnalRange)
        df_x.set_index(0,inplace=True)
        df_adc = df_in.append(df_x) # This merges the two indexes 
        df_adc = df_adc.loc[~df_adc.index.duplicated(keep='first')] #Keep first because real data is first on the list
        df_adc.sort_index(inplace=True) # this is sorted with intervening nans that need to be imputtecd/intrpolated
        df_adc_int = df_adc.interpolate(method='linear')
        self.df_adc_wl = df_adc_int.loc[diurnalRange]
        #
        df_in= self.df_obs_wl.copy()
        df_x = pd.DataFrame(diurnalRange)
        df_x.set_index(0,inplace=True)
        df_obs = df_in.append(df_x) # This merges the two indexes 
        df_obs = df_obs.loc[~df_obs.index.duplicated(keep='first')] #Keep first because real data is first on the list
        df_obs.sort_index(inplace=True) # this is sorted with intervening nans that need to be imputtecd/intrpolated
        df_obs_int = df_obs.interpolate(method='linear')
        self.df_obs_wl = df_obs_int.loc[diurnalRange]
        utilities.log.info('ADC and OBS have been corrected (inplace) for Diurnal periodicity')
 
# Many edge cases can mess this procedure up
    def _computeAndAverageErrors(self):
        """We expect that time bounds have been set so we can average over period in multiple of 12 steps
        We also expect semidiurnal range to have been corrected for
        We compute these errors based on interpolated (semi) Diurnal adc and obs data sets
        NOTE: at this stage self.df_adc and self.df_obs are in the semidiurnal transformed time basis
        """
        stripNans=False
        self.diff = self.df_adc_wl - self.df_obs_wl
        ##
        ## Look for outliers now and construct adc/obs moving forward as needed
        self.df_merged = self._combineDataFrames(self.df_adc_wl.copy(), self.df_obs_wl.copy(), self.diff.copy())
        diff = self.diff.copy()  # We want to manipulate inplace so do not disturb the real errro object
        # NANs at this level are prob coming foem the ADC data set
        # We want the results to be in REVERSE order with oldest subaverage first)
        # diff.sort_index(ascending=False, inplace=True) #  No need to reverse indexing
        diff.reset_index(inplace=True) # Specify integers as indexes so we can use a groupby function
        utilities.log.info('Averaging: groupby '+str(self.n_period))
        if self.n_period != 12:
            sys.exit('Averaging: n_periods not tested beyond = 12')
        self.df_cycle_avgs = diff.groupby(diff.index // self.n_period).mean().T
        self.df_cycle_avgs.columns = ['Period_'+str(x) for x in self.df_cycle_avgs.columns]
        self.df_cycle_avgs = pd.concat([self.df_meta[['lon','lat','Node']], self.df_cycle_avgs],axis=1)
        # Now get fullmean and std without period boundaries
        self.df_final = pd.DataFrame([diff.drop('index',axis=1).mean(), diff.drop('index',axis=1).std()]).T
        self.df_final.columns=['mean','std']
        self.df_final = pd.concat([self.df_meta[['lon','lat','Node']], self.df_final],axis=1)
        # Merge lon,lat values into the final data product
        # Must remove stations in the final and period resulots that have nans
        if stripNans:
            utilities.log.info('Removing potential NANS from summary data')
            if self.df_final.isnull().any().sum() != 0:
                utilities.log.info('Nans found in the summary means: those stations will be removed')
                print('Original summary size was '+str(self.df_final.shape))
                self.df_final.dropna(axis=0, inplace=True) # Only if ALL columns are nan
                print('Post summary size was '+str(self.df_final.shape))
            if self.df_cycle_avgs.isnull().any().sum() != 0:
                print('Original cycle size was '+str(self.df_cycle_avgs.shape))
                utilities.log.info('Nans found in the periodic means: those stations will be removed')
                self.df_cycle_avgs.dropna(axis=0,inplace=True) # Only if ALL columns are nan
                print('Post cycle size was '+str(self.df_cycle_avgs.shape))
        if self.config['ERRORFIELD']['EX_OUTLIER']:
            utilities.log.info('Check for station outliers')
            drop_stationlist = self._findStationOutliers()
            ## Remove from all summary and period statistics
            utilities.log.info('Removing outlier stations from summary statistics')
            utilities.log.info('List of stations removed is {}.'.format(str(drop_stationlist)))
            self.df_final.drop(drop_stationlist,inplace=True) 
            self.df_cycle_avgs.drop(drop_stationlist,inplace=True)
            utilities.log.info('Removing outlier stations from ADC, OBS and ERR, time data')
            self.df_adc_wl.drop(drop_stationlist,axis=1,inplace=True)
            self.df_obs_wl.drop(drop_stationlist,axis=1,inplace=True)
            self.diff.drop(drop_stationlist,axis=1,inplace=True)
            self.df_merged.drop(drop_stationlist,axis=1,inplace=True)
            print(str(self.diff.shape))
            #print(self.diff)
        
    def _combineDataFrames(self,adc,obs,err):
        """Combines the three diurnal corrected data frames into a single multiindex object
        """
        # Add a new column for building index
        adc['SRC']='ADC'
        obs['SRC']='OBS'
        err['SRC']='ERR'
        # Specify current index name
        adc.index.name='TIME'
        obs.index.name='TIME'
        err.index.name='TIME'
        # Do the work
        adc.reset_index(inplace=True)
        obs.reset_index(inplace=True)
        err.reset_index(inplace=True)
        adc.set_index(['TIME','SRC'], inplace=True)
        obs.set_index(['TIME','SRC'], inplace=True)
        err.set_index(['TIME','SRC'], inplace=True)
        df_merged = pd.concat([adc,obs,err])
        return df_merged

    def _outputDataToFiles(self, metadata='_test',subdir='errorfield'):
        """Dump the averages to disk for posterity
        """
        #print(self.df_final)
        #print(self.df_cycle_avgs)
        print('writing the files')
        df_metadata = pd.DataFrame([self._constructMetaData()])
        self.finalfilename=utilities.writeCsv(self.df_final,rootdir=self.rootdir,subdir=subdir,fileroot='stationSummaryAves',iometadata=metadata)
        self.cyclefilename=utilities.writeCsv(self.df_cycle_avgs,rootdir=self.rootdir,subdir=subdir,fileroot='stationPeriodAves',iometadata=metadata)
        self.metafilename=utilities.writeCsv(df_metadata,rootdir=self.rootdir,subdir=subdir,fileroot='stationMetaData',iometadata=metadata)
        self.mergedname=utilities.writeCsv(self.df_merged,rootdir=self.rootdir,subdir=subdir,fileroot='adc_obs_error_merged',iometadata=metadata)
        self.errorfilename=utilities.writePickle(self.diff,rootdir=self.rootdir,subdir=subdir,fileroot='tideTimeErrors',iometadata=metadata)
        self.jsonfilename=utilities.writeJson(self.df_merged.reset_index(),rootdir=self.rootdir,subdir=subdir,fileroot='adc_obs_error_merged',iometadata=metadata)
        #self.df_merged.reset_index().to_json(self.jsonfilename)
        utilities.log.info('save averaging files ' + self.finalfilename +' and '+ self.cyclefilename +' and '+self.metafilename +' and '+self.errorfilename +' and '+ self.mergedname +' and '+ self.jsonfilename) 

    def _fetchOutputFilenames(self):
        return self.errorfilename, self.finalfilename, self.cyclefilename, self.metafilename, self.mergedname, self.jsonfilename

    def executePipeline(self, metadata = 'Nometadata',subdir=''):
        dummy = self._intersectionStations()
        dummy = self._intersectionTimes()
        dummy = self._tidalCorrectData()
        dummy = self._applyTimeBounds()
        dummy = self._computeAndAverageErrors()
        dummy = self._outputDataToFiles(metadata=metadata,subdir=subdir)
        errf, finalf, cyclef, metaf, mergedf, jsonf = self._fetchOutputFilenames()
        ##dummy = self._generatePerStationPlot(metadata='Nometadata')
        return errf, finalf, cyclef, metaf, mergedf, jsonf

    def executePipelineNoTidalTransform(self, metadata = 'Nometadata',subdir=''):
        dummy = self._intersectionStations()
        dummy = self._intersectionTimes()
        #dummy = self._tidalCorrectData()
        dummy = self._applyTimeBounds()
        dummy = self._computeAndAverageErrors()
        dummy = self._outputDataToFiles(metadata=metadata,subdir=subdir)
        errf, finalf, cyclef, metaf, mergedf, jsonf = self._fetchOutputFilenames()
        return errf, finalf, cyclef, metaf, mergedf, jsonf

# Combine these intermediate steps into a single caller
def main(args):
    # print (args)
    meta = args.obsmeta
    obsf = args.obsdata
    adcf = args.adcdata
    extraExpDir = args.extraExpDir if args.extraExpDir!=None else 'ComputeError'
    missingFiles=True
    if (meta!=None) and (obsf!=None) and (adcf!=None):
        missingFiles=False

    # We can change the class to use the load_config() eliminating the need for this yaml ref.
    config = utilities.load_config()
    #extraExpDir='TestComputeError'
    #extraExpDir=''
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'],basedirExtra=extraExpDir)
    #
    #dir='/home/jtilson/ADCIRCDataAssimilation/test/errorfield/'
    #dir='../test/errorfield/'
    if missingFiles:
        dir = os.path.dirname(__file__)+'/ADCIRC'
        utilities.log.info('1 or more inputs files missing Try instead to find in dir {}'.format(dir))
        meta='/'.join([dir,'obs_wl_metadata.pkl'])
        obsf='/'.join([dir,'obs_wl_smoothed.pkl'])
        adcf='/'.join([dir,'adc_wl.pkl'])
    print('Run computeErrorField')
    cmp = computeErrorField(obsf, adcf, meta, rootdir=rootdir, aveper=4)
    dummy = cmp._intersectionStations()
    dummy = cmp._intersectionTimes()
    #dummy = cmp._tidalCorrectData()
    dummy = cmp._applyTimeBounds()
    dummy = cmp._computeAndAverageErrors()
    dummy = cmp._outputDataToFiles(metadata='_maintest',subdir='') # Note the delimiter is added here
    print('get output names')
    errf, finalf, cyclef, metaf, mergedf, jsonf = cmp._fetchOutputFilenames()
    print('output files '+errf+' '+finalf+' '+cyclef+' '+metaf+' '+mergedf+' '+jsonf)

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('--obsmeta', action='store', dest='obsmeta',default=None, help='FQFN to obs_wl_metadata.pkl', type=str)
    parser.add_argument('--obsdata', action='store', dest='obsdata',default=None, help='FQFN to obs_wl_smoothed.pkl', type=str)
    parser.add_argument('--adcdata', action='store', dest='adcdata',default=None, help='FQFN to adc_wl.pkl', type=str)
    parser.add_argument('--extraExpDir', action='store', dest='extraExpDir', default=None, help='Subdir to store files', type=str)
    args = parser.parse_args()
    sys.exit(main(args))

