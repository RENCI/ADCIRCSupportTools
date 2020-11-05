#!/usr/bin/env python

# Class to manage fetching water levels from the NOAA COOPS servers
# Note though untested, in theory one could ask for any of the available products. But we
# are only interested in water_levels at this time
#
# Management of missingness in the stations or its data.
# Several missingness selections can be provided to the YAML file used by the job.
# These include EX_COOPS_NANS, EX_MULTIVALUE, EX_THRESH, and THRESH.
# EX_COOPS_NANS. must be True at this time: If a call to the coops server for station metadata
#    return nothing then that station doesn't exist. It gets permanantly removed.
# EX_MULTIVALUE. Occasionally one or more time calls to a station can result in multiple
#    product levels. When this happends all duplicate times are removed and the FIRST
#    entry is kept. Setting EX_MULTIVALUE=True will additionaly cause this station to be permanently 
#    removed as a suspect station
# EX_THRESH and THRESH work together to manage stations with partial missingness in water levels.
#    if EX_THRESH=True, then stations with a loss percentage >= THRESH (%) will be
#    permanently excluded
# TODO An outlier detection procedure
# Recommended values: EX_COOPS_NANS=True, EX_MULTIVALUE=False, EX_THRESH=True, THRESH=10.0
#
# Internally, time formats vary between actual pd.Timestamps and strings
# We still allow the caller
# to provide the desired date as a string in the format yy-mm-dd(T)HH:MM:SS.
#
# Product data is now returned as a df (date_time x stations) in a 2D format. 
# A new dataframe carries the number of nans(and percentages) to allow subsequent exclusions 
# based on stations.
#
# Outputs: several files are (optionally) written to disk
# All filename selections can be customized using calling-program supplied metadata
#    For illustration, filenames generated using the metadata "metadata" are listed.
#    obs_wl_detailed_metadata.pkl: Time x Station product level data at 6 (sometimes 1) min intervals
#    obs_wl_smoothed_metadata.pkl: Time x Station product level at an hourly rate
#        Data resulting from window=11, interpolation. Will remove last nans in the data set
#        The interpolation will results in nans at the head/tail of the time range - these
#            get replaced with actual values  
#        Optionally call a savgol_filter though it is less tested
#    obs_wl_exclude_metadata.cvs: List of excluded stations and reason for exclusion
#    obs_wl_urls_metadata.cvs: List of URLs for finding NOAA-COOPS pages with proper time range
#        for all stations ( included and excluded) in the input list
#    obs_wl_metadata_metadata.pkl: Lon,Lat,Name,StationsID,ADCIRC nodeid for retained stations
#
# Generally all we really want are the smoothed hourly data 

# TODO excludeStationList is written out by the URL method. This needs to be refactored

import os, sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from scipy.signal import savgol_filter
import webbrowser
import urllib.parse
from utilities.utilities import utilities
import noaa_coops as coops
from requests.exceptions import ConnectionError,Timeout,HTTPError

# globals: Note for now we will only tested MSL .
# https://tidesandcurrents.noaa.gov/api/#products
datums = ('CRD', 'IGLD', 'LWD', 'MHHW', 'MTL', 'MSL', 'MLW', 'MLLW', 'NAVD', 'STND')
products = ('water_level', 'air_temperature', 'water_temperature', 'wind', 'air_pressure',
            'air_gap', 'conductivity', 'visibility', 'humidity', 'salinity', 'hourly_height',
            'high_low', 'daily_mean', 'monthly_mean', 'one_minute_water_level', 'predictions',
            'datums', 'currents')
timezones = ('gmt', 'lst', 'lst_ldt')
units = ('metric', 'english')

# If you leave off the %M, term noaa-coops will error with format error
def processToStationFormat(timein, timeout):
    """ 
    NOTE as of Jan 13, the noaa_coops website indicates we should be able to use other 
    formats. But my testing testing indicates not

    Parameters:
        timein, timeout: Input times as timestamps.
    Returns:
        timein, timeout in string format.
    """
    utilities.log.info("processToStationFormat is only %Y%m%d %H:%M for now")
    return timein.strftime('%Y%m%d %H:%M'), timeout.strftime('%Y%m%d %H:%M')

def applyTimeBoundaries(timein, timeout, single_stationdata):
    """
    noaa_coops returns an hard to predict number of data points between broad times of yyyymmdd. 
    So here we extract only those data product date_time values that fall within the fine grained
    boundaries specified by the user. 

    Parameters:
        timein, timeout: timestamps range of products requested to keep.
        single_stationdata: dataframe (time x product) Current product timeseries for the indicated station.
    Returns:
        New timeseries (dataframe) retaining only times within the input range.
    """
    if not isinstance(timein, pd.Timestamp) or not isinstance(timeout, pd.Timestamp):
        utilities.log.error("input time ranges should be pd.timestamp:".format(type(timein)))
        sys.exit(1)
    df = single_stationdata
    lower = df['date_time'] >= timein 
    upper = df['date_time'] <= timeout
    return df[lower & upper]

def convertToTimeStamp(timein, timeout):
    """
    Here we ensure that the final timestamp form is pd.Timestamp.
    The user may input this as str, numpy.dateframe64 or pd.timestamp. 

    Parameters:
        timein, timeout: str, pd.Timestamp, datetime64, or np.timestamp.
    Returns:
        timein, timeout: pd.Timestamp.
    """
    return pd.Timestamp(timein), pd.Timestamp(timeout)

class GetObsStations(object):
    """ 
    Class to establish connection to the noaa_coops servers and acquire a range of product levels for the set
    of stations.
    Input station IDs must be stationids and they are treated as string values.

    Parameters: 
        datum: ('MSL').
        unit: ('metric').
        timezone: ('gmt').
        product: ('water_level').
        metadata ('Nometadata'): Applied to all output filenames to create user-tagged names
        Eg obs_wl_urls_metadata.cvs'.
        rootdir ('None')(Mandatory): Locates finakl storage. Files are stored in rootdir/'obspkl'
        and '.' is a valid option.
        config['OBSERVATIONS']['EX_COOPS_NANS'].
        config['OBSERVATIONS']['EX_MULTIVALUE'].
        config['OBSERVATIONS']['EX_THRESH'].
        float(config['OBSERVATIONS']['THRESH']).
    """
    def __init__(self, datum='MSL', unit='metric', product='water_level', yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml'),timezone='gmt', metadata='',rootdir='None', iosubdir='obspkl'):
        """
        get_obs_stations constructor

        Parameters: 
            datum: str, product reference.
            unit: str, product units.
            timezone, str.
            product: str, prodeuct type desired.
            metadata: str, filename extra naming.
            rootdir: str, high level output directory.
            iosubdir: Subdirectory under trootdir
        """
        self.excludeStationID = pd.DataFrame() # will carry the following stationid (index), enum(missing,nan,outlier)
        self.stationlist = list()
        self.datum = datum.upper() if datum.upper() in datums else 'None'
        self.unit = unit.lower() if unit.lower() in units else 'None'
        self.product = product.lower() if product.lower() in products else 'None'
        self.timezone = timezone.lower() if timezone.lower() in timezone else 'None'
        self.iometadata = metadata 
        self.rootdir = rootdir
        self.iosubdir = iosubdir
        if self.rootdir == None:
            utilities.log.error("No rootdir specified on init {}".format(self.rootdir))
            sys.exit(1)
        self.config = utilities.load_config(yamlname) 
        self.ex_coops_nans = self.config['OBSERVATIONS']['EX_COOPS_NANS']
        self.ex_multivalue = self.config['OBSERVATIONS']['EX_MULTIVALUE']
        self.ex_thresh = self.config['OBSERVATIONS']['EX_THRESH']
        self.nanthresh = float(self.config['OBSERVATIONS']['THRESH'])
        if not self.ex_coops_nans:
            utilities.log.error('EX_COOPS must be True for all nontrivial work')

    def fetchStationListIncludeAndExclude(self):
        """
        Combine all included and excluded stations into a single list
        for use by the URL methods.

        Parameters:
            None. 
        Results:
            list (str) combined stationIDs (included and excluded).
        """
        utilities.log.info("Fetching combined include exclude station list: may cause URL prob if keeping NOAA nan stations")
        excludeList = list(set(self.excludeStationID.index.values))
        includeList = self.stationlist
        return includeList+excludeList

    def stationListFromYaml(self):
        """
        Insert a list into the YAML
        Parameters:
            None
        Results:
            list (str) of stationIDs from the YAML file
        """
        stationList = self.config['OBSERVATIONS']['STATIONS'] # Just guessing here
        self.stationlist = stationList
        return stationList

    def fetchStationNodeList(self):
        """
        Grab all station IDs from the user supplied YAML file. In this YAML file 
        (optionally) is the name for the file that contains the stationID list.
        Format requirement apply and are not checked. This method is not needed
        for actual ADDA processing

        Parameters:
            None.
        Results:
            Return dataframe ('stationid','Node','lon','lat']). 
        """
        config = self.config['DEFAULT']
        stafile = os.path.join(os.path.dirname(__file__), "../config", config['StationFile'])
        try:
            # NOTE datafile format in flux Need to skip the 2nd row
            df = pd.read_csv(stafile, index_col=0, header=0, skiprows=[1], sep=',')
            stationNodelist = df[['stationid','Node','lon','lat']] # Which long/lat do you want
        except FileNotFoundError:
            raise IOerror("Failed to read %s" % (config['StationFile']))
        return stationNodelist

    def checkDuplicateTimeEntries(self, station, stationdata):
        """
        Sometimes station data comes back with multiple entries for a single time.
        Here we search for such dups and keep the LAST one
        Choosing first was based on a single station and looking at the noaa coops website

        Parameters:
            station: str an individual stationID to check
            stationData: dataframe. Current list of all station product levels (from detailedIDlist) 
        Results:
            New dataframe containing no duplicate values
            multivalue. bool: True if duplicates were found 
        """
        multivalue = False
        idx = stationdata.index
        if idx.duplicated().any():
            utilities.log.info("Duplicated Obs data found for station {} will keep first value(s) only".format(str(station)))
            stationdata = stationdata.loc[~stationdata.index.duplicated(keep='first')]
            multivalue = True
        return stationdata, multivalue

    def fetchStationMetaDataFromIDs(self, stationlist):
        """
        Read the yaml file. return the five features (IDs,name,lat,lon,Node) for the selected
        and validated stations. 
        The class list (stationlist) is updated INPLACE and returned. The excluded list is also
        captured for possible post analysis

        Parameters:
            stationlist: list of str stationIDs. Updates several class variables.
        Results:
            dataframe: if station metadfata 'stationid', 'stationname', 'lat', 'lon', 'Node'].
            stationlist: List (str) of retained stationIDs (dep on value for EX_COOPS_NANS).
        """
        #config = self.config['OBSERVATIONS']
        config = self.config

        tempstationlist = stationlist
        stafile = os.path.join(os.path.dirname(__file__), "../config", config['DEFAULT']['StationFile'])
        try:
            # NOTE datafile format in flux Need ot skip the 2nd row
            df = pd.read_csv(stafile, index_col=0, header=0, skiprows=[1], sep=',')
            # Are the user supplied stationids all available? If not keep valid ones and print a warning
            avail_df = df[df['stationid'].isin(stationlist)] # stationlist must be strings
            if avail_df.shape[0] != len(stationlist):
                utilities.log.info("Warning: the requested station IDs were not all found. Found ones will be kept")
                missing = [x for x in stationlist if x not in set(avail_df['stationid'])]
                temp_excludeStationID = pd.DataFrame(missing).set_index(0)
                temp_excludeStationID['STATUS']='EXCLUDE_Not_in_NOAA_list'
                self.excludeStationID = self.excludeStationID.append(temp_excludeStationID)
                tempstationist = self.stationlist
                if self.ex_coops_nans:
                    tempstationlist = [x for x in stationlist if x not in missing]
            self.stationlist = tempstationlist
            if len(self.stationlist) == 0:
                utilities.log.error("No valid stationIDs remain from NOAA COOPS call")
                # print('WARNING: No valid stationIDs remain: Inputs were ')
                # print(stationlist)
                sys.exit(1)
            df_stationData = df[df['stationid'].isin(self.stationlist)][
                ['stationid', 'stationname', 'lat', 'lon', 'Node']]
        except FileNotFoundError:
            raise IOerror("Failed to read %s" % (config['StationFile']))
        utilities.log.info("Final Metadata products")
        self.metapkl = utilities.writePickle(df_stationData,rootdir=self.rootdir,subdir=self.iosubdir,fileroot='obs_wl_metadata',iometadata=self.iometadata)
        self.metajsonname = utilities.getSubdirectoryFileName(self.rootdir, self.iosubdir, 'obs_wl_metadata'+self.iometadata+'.json')
        df_stationData.to_json(self.metajsonname)
        return df_stationData, self.stationlist

    def fetchStationProductFromIDlist(self, timein, timeout):
        """ 
        Fetch the selected data products. a timeseries of values for each stationID.  
        (for now this is only tested for water_level)
        return all values within (inclusive) the provided range.
        the time input to noaa_coops is HOURLY yyyymmdd while the user inputs a range in the format 
        yyyy-mm-dd hh:mm. So noaa-coops may return more than needed for the DA process.

        Parameters:
            timein, timeout: in str or timestanp format. The detailed time range (inclusive).
            to fetch product values.

        Results:
            dataframe: Time x station matrix in df format within the timein,timeout range. Some stations may
            be excluded depending on user selection for missingness (EX_MULTIVALUE and EX_THRESH).
            count_nan: dataframe of num nans, %nans, total vals for each station (Used for subsequent filtering).
            stationlist: List (str) of current set of validated stationIDs (also updates class list).
   
            Write out product list to disk
        """
        list_frame = list()
        exclude_stations = list()
        df_final = pd.DataFrame()
        timein, timeout = convertToTimeStamp(timein, timeout)
        station_timein, station_timeout = processToStationFormat(timein, timeout)
        utilities.log.info("station IDlist timein {} and timeout {}".format(timein, timeout))
        for station in self.stationlist:
            try:
                stationdata = pd.DataFrame()
                location = coops.Station(station)
                stationdata = location.get_data(begin_date=str(station_timein),
                                                end_date=str(station_timeout),
                                                product=self.product,
                                                datum=self.datum,
                                                units=self.unit,
                                                time_zone=self.timezone)[self.product].to_frame()
                stationdata, multivalue = self.checkDuplicateTimeEntries(station, stationdata)
                if self.ex_multivalue and multivalue:
                    utilities.log.info('Multivalued station '+str(station))
                    exclude_stations.append(station)
                else:
                    stationdata.reset_index(inplace=True)
                    stationdata = applyTimeBoundaries(timein, timeout, stationdata)
                    stationdata.set_index(['date_time'], inplace=True)
                    stationdata.columns=[station]
                    list_frame.append(stationdata)
            except ConnectionError:
                utilities.log.error('Hard fail: Could not connect to COOPS for water products {}'.format(station))
            except HTTPError:
                utilities.log.error('Hard fail: HTTP error to COOPS for water products')
            except Timeout:
                utilities.log.error('Hard fail: Timeout error to COOPS for water products')
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                exclude_stations.append(station)
        if len(exclude_stations) > 0:
            newlist = self.stationlist
            # print('Removing stations from list')
            utilities.log.info('Removing stations from stationlist: Number to remove is {}'.format(len(exclude_stations)))
            temp_excludeRepsStationID = pd.DataFrame(exclude_stations).set_index(0)
            temp_excludeRepsStationID['STATUS']='EXCLUDE_Removed_Multivalued_obs'
            self.excludeStationID = self.excludeStationID.append(temp_excludeRepsStationID)
        utilities.log.info("Length of stations for product fetch is {}.".format(str(len(list_frame))))
        df_final = df_final.join(list_frame, how='outer') # Preserves indexing and adds nans as required
        # POtentially Remove any multivalued stations in the exclude list
        #  df_final.drop(exclude_stations,axis=1,inplace=True)
        self.stationlist = df_final.columns.values.tolist()
        total_elems = len(df_final) 
        num_nans = total_elems - df_final.count()
        count_nan = num_nans.to_frame()
        count_nan['Number'] = total_elems
        count_nan['PercentNans']=num_nans*100.0/total_elems
        count_nan.columns=['NumNans','Total','Percent']
        count_nan.index.name='stationid'
        utilities.log.info("Final detailed data product: Number of times {}. Num Stations {}".format(df_final.shape[0],df_final.shape[1]))
        print('Final detailed data product: Number of times %i, Number of stations %i' % df_final.shape)
        utilities.log.info("Writing PKL for detailed 6min data")
        self.detailedpkl = utilities.writePickle(df_final, rootdir=self.rootdir,subdir=self.iosubdir,fileroot='obs_wl_detailed',iometadata=self.iometadata)
        self.detailedjsonname = utilities.getSubdirectoryFileName(self.rootdir, self.iosubdir, 'obs_wl_detailed'+self.iometadata+'.json')
        dfjson=df_final.copy()
        dfjson.index = dfjson.index.strftime('%Y-%m-%d %H:%M:%S')
        dfjson.to_json(self.detailedjsonname)
        return df_final, count_nan, self.stationlist, self.excludeStationID.index.tolist()

    def fetchOutputNames(self):
        """
        Results:
            detailedpkl, smoothpkl, metapkl, urlcsv, excludecsv meta-json, detailed-json, smoothed-json
        """
        return self.detailedpkl, self.smoothpkl, self.metapkl, self.urlcsv, self.excludecsv, self.metajsonname, self.detailedjsonname, self.smoothedjsonname

# On input, the caller should have self.removeMissingProducts() already
# As this method will interpolate through the nans
#
    def fetchStationSmoothedHourlyProductFromIDlist(self, timein, timeout, percentage_cutoff=None):
        """ 
        Fetch the selected data product (for now this is only tested for water_level)
        return all values within the provided range.
        The time input to noaa_coops is at yyyymmdd while the user inputs a range in the format yyyy-mm-dd hh:mm
        Get the detailed product levels then perform a smooth and then find which indexes to keep.
        For remaining hourly data their will still be nans because of finite window widths. Simply replace with 
        actual values from df_detailed
        Smoothing itself can introduce nans to a timeseries. For example in a rolling average. Filtering 
        on those nans is not performed. Rather, those nans are replaced with actual values from the
        detailed_list data set.

        Parameters:
            timein, timeout: str or timestamp. Hourly range to fetch product levels.
        Results:
            dataframe: smoothed Time x station matrix in df format within the timein,timeout range. Some stations may
            be excluded depending on user selection for missingness (EX_MULTIVALUE and EX_THRESH).
            count_nan: dataframe of num nans, %nans, total vals for each station (Used for subsequent filtering).
            stationlist: List (str) of current set of validated stationIDs (also updates class list).

            Write out smoothed data to disk
            Write out exluded list of stations to disk
        """
        df_detailed, count_nan, stationlist, excludelist = self.fetchStationProductFromIDlist(timein, timeout)
        df_det_filter, newstationlist, excludelist = self.removeMissingProducts(df_detailed, count_nan, percentage_cutoff=percentage_cutoff) # If none then read from yaml;
        #df_smoothed = self.smoothVectorProducts( df_detailed, window=11, degree=3 )
        df_smoothed = self.smoothRollingAveProducts( df_det_filter, window=11)
        df_smoothed = df_smoothed.loc[df_smoothed.index.strftime('%M:%S')=='00:00'] # Is this sufficient ?
        total_elems = len(df_smoothed)
        num_nans = total_elems - df_smoothed.count()
        count_nan = num_nans.to_frame()
        count_nan['Number'] = total_elems
        count_nan['PercentNans']=num_nans*100.0/total_elems
        count_nan.columns=['NumNans','Total','Percent']
        count_nan.index.name='stationid'
        utilities.log.info("Smoothed hourly data product: Number of times {}. Num Stations {}".format(df_smoothed.shape[0],df_smoothed.shape[1]))
        utilities.log.info("Writing PKL for Smoothed hourly data")
        self.smoothpkl = utilities.writePickle(df_smoothed, rootdir=self.rootdir,subdir=self.iosubdir,fileroot='obs_wl_smoothed',iometadata=self.iometadata)
        self.excludecsv = utilities.writeCsv(self.excludeStationID, rootdir=self.rootdir,subdir=self.iosubdir,fileroot='obs_wl_exclude',iometadata=self.iometadata)
        self.stationlist = newstationlist
        self.smoothedjsonname = utilities.getSubdirectoryFileName(self.rootdir, self.iosubdir, 'obs_wl_smoothed'+self.iometadata+'.json')
        dfjson=df_smoothed.copy()
        dfjson.index = dfjson.index.strftime('%Y-%m-%d %H:%M:%S')
        dfjson.to_json(self.smoothedjsonname)
        return df_smoothed, count_nan, stationlist, excludelist

    def smoothVectorProducts( self,  df_in, window=21, degree=3 ):
        """
        Not used for now.
        Apply a Savitzky-Golay filter.

        Parameters:
            df_in: input dataframe of times x stations.
            window: width (int) (12).
            degree: polynomial degree (3). 
        Results:
            df_smoothed: dataframe (time x stations) smoothed.
        """
        df_smooth = savgol_filter(df_in, window, degree)
        return df_smooth

    def smoothRollingAveProducts( self, df_in, window=11):
        """
        CENTERED window rolling average.

        Parameters:
            df_in: input dataframe of times x stations.
            window: (int,def=11) width of window.
        Results:
            df_smoothed: dataframe (time x stations) smoothed.
        """
        df_smooth = df_in.rolling(window=window, center=True).mean()
        indlist = df_smooth.loc[df_smooth.isnull().all(1)].index # Only if ALL columns are nan
        df_smooth.loc[indlist] = df_in.loc[indlist] # No worries: we winnow non hourlies later
        return df_smooth

    def setStationIDs(self, stationlist):
        """
        This method allows the user to override the current list of stations to process
        They will be subject to the usual validity testing. We overwrite any station data fetched in the class

        Parameters:
            stationlist: list (str) of stationIDs. Overrides any existing list.
        """
        df_stationData, self.stationlist = self.fetchStationMetaDataFromIDs(stationlist)
        return df_stationData, self.stationlist


# Update the excludeList but as a dataframe including metadata
# NOTE: Should be run only after invocation of the detailed ListID methods
    def removeMissingProducts(self, df_in, counts, percentage_cutoff=None):
        """
        Remove stations with nan rates >= input thresholda cutoff
        Must be invoked by the calling program: Should only be applied to
        DETAILED_DATA lists. For smoothed data we clean up data a different way
        Stations that are excluded also update the class list excludeStationID.

        Parameters:
            df_in: Input dataframe (time x stations).
            counts: dataframe: count_nan.columns=['NumNans','Total','Percent'].
            percentage_cutoff: float. Percent cutoff.
        Results:
            df_out: dataframe (time x stationID) with stations excluded.
            stationlist: list (str) of stationIDs.
        """
        # could be none
        if percentage_cutoff == None:
            percentage_cutoff = self.nanthresh # Grabs from config yaml
        if not self.ex_thresh:
            utilities.log.info('removeMissingness called by EX_THRESH set to False: Ignore')
            return df_in, self.stationlist
        exclude_stations = list(counts.loc[counts['Percent'] > percentage_cutoff].index.values)
        utilities.log.info("Removing {} stations based on a percent cutoff of {}.".format(len(exclude_stations), percentage_cutoff))
        utilities.log.info('Removing the following stations because of Max Nan %: '+str(exclude_stations))
        if len(exclude_stations) > 0:
            temp_excludeStationID = pd.DataFrame(exclude_stations).set_index(0)
            temp_excludeStationID['STATUS']='EXCLUDE_Max_NAN_threshold '+str(percentage_cutoff)
            self.excludeStationID = self.excludeStationID.append(temp_excludeStationID)
        df_out = df_in.drop(exclude_stations,axis=1)
        self.stationlist = df_out.columns.values.tolist()
        return df_out, self.stationlist, self.excludeStationID.index.tolist()

    def writeURLsForStationPlotting( self, liststations, timein, timeout ):
        """
        Rudimentary function to take a list of station IDs and build URLs with the proper
        time ranges, etc for calling noaa-coops and viewing their data directly. 
        This is really only meant for code validation and
        is not guaranteed to work in generalized environments.
        URLs are constructed for both included and excluded stations.

        Parameters:
            timein, timeout: time range (hourly for station data inspection.
        Results:
            URLs: written to file: obs_wl_urls_iometadata.cvs' in rootdir/obspkl/.
        """
        lens = len(liststations)
        station_timein = pd.Timestamp(timein).strftime('%Y%m%d')
        station_timeout = pd.Timestamp(timeout).strftime('%Y%m%d')
        d = []
        utilities.log.debug('Write URLs timein {} timeout {} station_timein {} station_timeout {}'.format(timein,
            timeout, station_timein, station_timeout))
        for station in liststations:
            data = {}
            data['id']=str(station)
            data['units']=self.unit
            data['bdate']=station_timein
            data['edate']=station_timeout
            data['timezone']=self.timezone
            data['datum']=self.datum
            data['interval']=6
            data['action']=' '
            url_values=urllib.parse.urlencode(data)
            url = 'https://tidesandcurrents.noaa.gov/waterlevels.html'
            full_url = url +'?' +url_values
            new = 2
            d.append((station, full_url, new))
        df_url = pd.DataFrame(d,columns=['station','noaa_url','new']) 
        df_url.set_index('station',inplace=True)
        self.urlcsv = utilities.writeCsv(df_url, rootdir=self.rootdir,subdir=self.iosubdir,fileroot='obs_wl_urls',iometadata=self.iometadata)
        utilities.log.debug("Test class-base url code: Not for production use")

    def executeBasicPipeline(self, timein, timeout):
        """
        Combine basic steps into a single call"
        """
        print('Invoking OBS pipeline')
        utilities.log.info("ProductLevel Working in {}.".format(os.getcwd()))
        timein = pd.Timestamp(timein)
        timeout = pd.Timestamp(timeout)
    
        metadata = self.iometadata
        rootdir=self.rootdir
    
        rpl = GetObsStations(rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml'), metadata=metadata) 
        df_stationNodelist = rpl.fetchStationNodeList() 
        stationNodelist = df_stationNodelist['stationid'].to_list()
        #Fetch metadata abnd remove stations from the list that did not exists in coops
        df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stationNodelist)
    
        # Fetch the smoothed hourly data    
        df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
        retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
        # For now these two functions MUST be performed and in this order
        listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)
    
        # Dump some information
        detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson = rpl.fetchOutputNames()
        utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} META Json {}, Detailed Json {}, Smoothed Json {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson))
        return detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson

def executeBasicPipeline(rootdir, timein, timeout, metadata=''):
    """
    Combine basic steps into a single call"
    """
    print('Invoking OBS pipeline')
    utilities.log.info("ProductLevel Working in {}.".format(os.getcwd()))
    timein = pd.Timestamp(args.timein)
    timeout = pd.Timestamp(args.timeout)

    config = utilities.load_config() # Defaults to main.yml as sapecified in the config
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')

    rpl = GetObsStations(rootdir=rootdir, yamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml'), metadata=metadata) 
    df_stationNodelist = rpl.fetchStationNodeList() 
    stationNodelist = df_stationNodelist['stationid'].to_list()
    # Add a fake node
    # stationNodelist = stationNodelist + [999999]

    #Fetch metadata abnd remove stations from the list that did not exists in coops
    df_stationData, stationNodelist = rpl.fetchStationMetaDataFromIDs(stationNodelist)

    # Fetch the smoothed hourly data    
    df_pruned, count_nan, newstationlist, excludelist = rpl.fetchStationSmoothedHourlyProductFromIDlist(timein, timeout)
    retained_times = df_pruned.index.to_list() # some may have gotten wacked during the smoothing`
# For now these two functions MUST be performed and in this order
    listSuspectStations = rpl.writeURLsForStationPlotting(newstationlist, timein, timeout)

# Dump some information
    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson = rpl.fetchOutputNames()
    utilities.log.info('Wrote Station files: Detailed {} Smoothed {} Meta {} URL {} Excluded {} META Json {}, Detailed Json {}, Smoothed Json {}'.format(detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson))
    print('Finished with OBS pipeline')
    return detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metaJson, detailedJson, smoothedJson

def main(args):
    """
    A simple main method to demonstrate the usew of this class
    Only ../config/main.yaml file is required
    Some extra steps are included (such as adding/removing stations
    to demonstrate their use
    """
    utilities.log.info("ProductLevel Working in {}.".format(os.getcwd()))
    timein = pd.Timestamp(args.timein)
    timeout = pd.Timestamp(args.timeout)

    config = utilities.load_config() # Defaults to main.yml as sapecified in the config
    rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='StationTest')
    # NOTE: we add a presumtive delimited to our metadata. That way we can send a black
    iometadata = '_'+timein.strftime('%Y%m%d%H%M')+'_'+timeout.strftime('%Y%m%d%H%M')
    iometadata=''

    detailedpkl, smoothedpkl, metapkl, urlcsv, exccsv, metajson,detailedjson,smoothedjson = executeBasicPipeline(rootdir, timein, timeout, metadata=iometadata)
    utilities.log.info('Finished')

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('--timein', default="'2020-08-01 12:00'", help='Timein string', type=str)
    parser.add_argument('--timeout', default="'2020-08-05 12:00'", help='Timeout string', type=str)
    args = parser.parse_args()
    sys.exit(main(args))
