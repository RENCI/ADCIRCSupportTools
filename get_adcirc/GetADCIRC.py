#!/usr/bin/env python

##
## Bolt in the code for choosing directories setting subdir to ''
## To mimic existing behavior
## Folded get_water_levels63 and 61 into the class
##

import os
import datetime as dt
from datetime import timedelta
from datetime import datetime
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utilities.utilities import utilities
from utilities import CurrentDateCycle as cdc
from siphon.catalog import TDSCatalog

# noinspection PyPep8Naming,DuplicatedCode
def main(args):

    # Possible override YML defaults
    indtime1 = args.timein
    indtime2 = args.timeout
    indoffset = args.doffset
    iometadata = args.iometadata
    iosubdir = args.iosubdir if args.iosubdir!=None else 'ADCIRC'
    iometadata = args.iometadata if args.iometadata!=None else ''
    variableName = args.variableName
    writeJson = args.writeJson
    adcyamlfile=args.adcYamlname

    config = utilities.load_config() # Get main comnfig. RUNTIMEDIR, etc
    rootdir = utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra=iosubdir)
    utilities.log.info("Working in {}.".format(os.getcwd()))

# get station-to-node indices for grid
    obsyamlname=os.path.join(os.path.dirname(__file__), '../config', 'obs.yml')
    obs_utilities = utilities.load_config(obsyamlname)
    station_df = utilities.get_station_list()
    station_ids = station_df["stationid"].values.reshape(-1,)
    node_idx = station_df["Node"].values

    urljson = args.urljson
    if urljson is not None:
        if not os.path.exists(args.urljson):
            utilities.log.error('urljson file not found.')
            sys.exit(1)
        urls = utilities.read_json_file(args.urljson)
        utilities.log.info('Explicit JSON URLs provided')

    # get class instance
    #adc = Adcirc(dtime1=indtime1, dtime2=indtime2, doffset=indoffset, metadata=iometadata)
    
    if adcyamlfile==None:
        adcyamlfile = os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')
    utilities.log.info('Choose a config file of the name {}'.format(adcyamlfile))
    adc = Adcirc(dtime1=indtime1, dtime2=indtime2, doffset=indoffset, metadata=iometadata,yamlname=adcyamlfile)

    if urljson is None:
        # Then generate URL list from time ranges
        #adc.set_times(doffset=args.doffset)
        adc.set_times( adc.dtime1, adc.dtime2, adc.doffset)
        utilities.log.info("T1 (start) = {}".format(adc.T1))
        utilities.log.info("T2 (end)   = {}".format(adc.T2))
        adc.get_urls()
        utilities.log.info("List of available urls from time conditions:")
    else:
        adc.urls = urls
        utilities.log.info("List of available urls input specification:")

    # get grid coords for later interpolation
    # Not needed here
    adc.get_grid_coords()
    utilities.log.debug(' gridx {}'.format(adc.gridx[:]))
    utilities.log.debug(' gridy {}'.format(adc.gridy[:]))
    adcx = adc.gridx[:].tolist()
    adcy = adc.gridy[:].tolist()
    adc_coords = {'lon':adcx, 'lat':adcy}
        # df = get_water_levels63(adc.urls, node_idx, station_ids)
    ADCdir = rootdir
    ADCfile = ADCdir+'/adc_wl'+iometadata+'.pkl'
    ADCfilecords = ADCdir+'/adc_coord'+iometadata+'.json'
    if adc.config["ADCIRC"]["fortNumber"] == "63":
        if args.ignore_pkl or not os.path.exists(ADCfile):
            utilities.log.info("Building model matrix from fort.63.nc files...")
            ##df = get_water_levels63(adc.urls, node_idx, station_ids)
            df = get_water_levels63(adc.urls, node_idx, station_ids)
            df = df.sort_index()
            df.to_pickle(ADCfile)
            utilities.write_json_file(adc_coords, ADCfilecords)
        else:
            utilities.log.info(ADCfile+" exists.  Using that...")
            df = pd.read_pickle(ADCfile)
    else:
        # df = get_water_levels61(adc.urls, station_ids)
        print("Currently, only fort.63.ncs can be processed.")
        sys.exit(1)

    if writeJson:
        jsonfilename=writeToJSON(df, rootdir, iometadata, variableName=variableName)
        utilities.log.info('Wrote ADC WL as a JSON {}'.format(jsonfilename))

    # df.drop_duplicates(inplace=True)
    # if adc.config["DEFAULT"]["GRAPHICS"]:
    #     ax = df.iloc[:, 0:1].plot(marker='.', grid=True)
    #     ax.set_xlim(adc.T1-timedelta(days=1), adc.T2)  # pd.Timestamp('2020-02-02T07'), pd.Timestamp('2020-02-04T12'))
    #     ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    #     ax.xaxis.set_major_locator(mdates.DayLocator())
    #     plt.draw()
    # 
    #     ax = df.iloc[:, 0:3].plot(legend=False, grid=True)
    #     ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    #     # ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %d %H %Y'))
    #     # ax.set_xlim(adc.T1-timedelta(days=1), adc.T2) # pd.Timestamp('2020-02-02T07'), pd.Timestamp('2020-02-04T12'))
    #     ax.xaxis.set_major_locator(mdates.DayLocator())
    #     plt.draw()

# TODO rename doffset to dlag
class Adcirc:
    def __init__(self, metadata='', dtime1=None, dtime2=None, doffset=None, yamlname=os.path.join(os.path.dirname(__file__), '../config', 'adc.yml')):
        """
        Parameters: 
            datum: str, product reference.
            unit: str, product units.
            timezone, str.
            product: str, prodeuct type desired.
        """
        self.datecycle = cdc.CurrentDateCycle().datecycle
        self.config = utilities.load_config(yamlname)
        self.urls = {}
        self.T1 = ()
        self.T2 = ()
        self.gridx = ()
        self.gridy = ()
        self.dtime1 = dtime1
        self.dtime2 = dtime2
        self.doffset = doffset
        self.iometadata = metadata
        self.set_times(dtime1=self.dtime1, dtime2=self.dtime2, doffset=self.doffset)
        self.config = utilities.load_config(yamlname)


    def orig_set_times(self, dtime1=None, dtime2=None, doffset=None):
        """
        """
        tim = self.config["TIME"]
        tT2=tim["T2"]
        tT1=tim["T1"]
        if dtime2 is not None:
             self.T2 = dt.datetime.strptime(dtime2, '%Y-%m-%d %H')
        else:
            self.T2 = self.datecycle + dt.timedelta(days=tT2) # Now + lag on t2
        if dtime1 is not None:
            self.T1 = dt.datetime.strptime(dtime1, '%Y-%m-%d %H')

        else:
            if doffset is not None:
                if int(doffset >0):
                    utilities.log.info('Input doffset was specifed as > 0: {}'.format(doffset))
                    #sys.exit('Doffset must be <=')
                    self.T2 = self.T1 + dt.timedelta(days=int(doffset))
                else:
                    self.T1 = self.T2 + dt.timedelta(days=int(doffset))
            else:
                self.T1 = self.T2 + dt.timedelta(days=tT1)
        return self

    def set_times(self, dtime1=None, dtime2=None, doffset=None):
        """
        Modified to allow more complex time conditions.
        doffset is the number of days (+ or -) from the input time pivot 
        """
        # These are for backup defauilt seetings and may not be used
        # Both entries are offsets one for T1 asnd one for T2. T2 is generally zero
        tim = self.config["TIME"] # These are for backup defauilt seetings and may not be used
        tT2=tim["T2"]
        tT1=tim["T1"]
        if dtime2 is not None and dtime1 is not None : # Set times and return
            self.T2 = dt.datetime.strptime(dtime2, '%Y-%m-%d %H:%M')
            self.T1 = dt.datetime.strptime(dtime1, '%Y-%m-%d %H:%M')
            return self             
        if dtime2 is not None and dtime1 is None: # Step back doffset days to find dtime1 
            self.T2 = dt.datetime.strptime(dtime2, '%Y-%m-%d %H:%M')
            doff = doffset if doffset is not None else tT1 # doff must be < 0
            if doff > 0:
                utilities.log.error(' doffset must be < 0 when specifying timeout: Got {}'.format(doff))
            self.T1 = self.T2 + dt.timedelta(days=doff) 
            return self 
        if dtime2 is None and dtime1 is not None: # Step forward doffset days to find dtime2
            self.T1 = dt.datetime.strptime(dtime1, '%Y-%m-%d %H:%M')
            doff = doffset if doffset is not None else tT1 # doff must be > 0
            if doff < 0:
                utilities.log.error(' doffset must be > 0 when specifying timein: Got {}'.format(doff))
            self.T2 = self.T1 + dt.timedelta(days=doff) # doffset must be > 0
            return self  # Now consider edge cases
        # The following should be orig ADDA behavior
        if dtime2 is None and dtime1 is None : # Set dtime2 as NOW+tT2 and back off doffsets for dtime1
            self.T2 = self.datecycle + dt.timedelta(days=tT2)
            doff = doffset if doffset is not None else tT1 # 
            self.T1 = self.T2 + dt.timedelta(days=doff)
            return self 

    def get_grid_coords(self):
        """
        Gets the lon/lats of the ADCIRC grid
        :param url: THREDDS url pointing to an ADCIRC grid
        :return: gridx, gridy as attributes to the class instance
        """
        firsturl = first_true(self.urls.values())
        nc = nc4.Dataset(firsturl)
        self.gridx = nc.variables['x']
        self.gridy = nc.variables['y']

    def get_urls(self):
        """
        Gets a dict of URLs for the time range and other parameters from the specified
        THREDDS server, using siphon. Parameters are specified in the adc.yml config file

        :return: dict of datecycles ("YYYYMMDDHH") entries and corresponding url.
        """
        cfg = self.config["ADCIRC"]
        urls = {}   # dict of datecycles and corresponding urls
        maincat = cfg["baseurl"] + cfg["catPart"]
        cat = TDSCatalog(maincat)
        available_dates = cat.catalog_refs
        # filter available dates between T1 and T2, inclusive
        dates_in_range = np.array([])
        for i, d in enumerate(available_dates):
            date_time_obj = dt.datetime.strptime(d, "%Y%m%d%H")
            if self.T1 <= date_time_obj <= self.T2:
                utilities.log.debug("%02d : %s in range." % (i, date_time_obj))
                dates_in_range = np.append(dates_in_range, date_time_obj)
        dates_in_range = np.flip(dates_in_range, 0)
        # get THREDDS urls for dates in time range
        for i, d in enumerate(dates_in_range):
            dstr = dt.datetime.strftime(d, "%Y%m%d%H")
            url = cfg["baseurl"] + \
                  cfg["dodsCpart"] % ("2020",
                                      dstr,
                                      cfg["AdcircGrid"],
                                      cfg["Machine"],
                                      cfg["Instance"],
                                      cfg["fortNumber"])
            try:
                nc = nc4.Dataset(url)
                # test access
                z = nc['zeta'][:, 0]
                url2add = url
            except:
                utilities.log.info("Could not access {}. It will be skipped.".format(url))
                url2add = None
            urls[d] = url2add
        self.urls = urls

def writeToJSON(df, rootdir, iometadata, fileroot='adc_wl', variableName=None):
    utilities.log.info('User requests write data also as a JSON format')
    df.index.name='TIME' # Need to adjust this for how the underlying DICT is generated
    merged_dict = utilities.convertTimeseriesToDICTdata(df, variables=variableName)
    jsonfilename=utilities.writeDictToJson(merged_dict,rootdir=rootdir,subdir='',fileroot=fileroot,iometadata=iometadata)
    utilities.log.info('Wrote ADC Json as {}'.format(jsonfilename))
    return jsonfilename

def get_water_levels63(urls, nodes, stationids):
    """
    Retrieves ADCIRC water levels from nowcast URLS, using fort.63.nc files
    
    Parameters:
        urls: dict of THREDDS urls to get ADCIRC wl from
        nodes: list of nodes in ADCIRC grid that correspond to the stations
        stationids: NOAA NOS station ids, to label dataframe columns (the get_obs function
    Results:
        returns a similar dataframe with the same column labels
        DataFrame of water ADCIRC water levels at station/node locations
    """
    # generate an empty dataframe
    df = pd.DataFrame(columns=stationids)
    initialtime = None
    finaltime = None # Update class times to reflect actual data set fetched

    for datecyc, url in urls.items():
        utilities.log.info("{} : ".format(datecyc))
        if url is None:
            utilities.log.info("   Skipping. No url.")
        else:
            utilities.log.info("   Loading ... ")
            utilities.log.debug(url)
            nc = nc4.Dataset(url)

            # we need to test access to the netCDF variables, due to infrequent issues with
            # netCDF files written with v1.8 of HDF5.

            if "zeta" not in nc.variables.keys():
                utilities.log.error("zeta not found in netCDF for {}.".format(datecyc))
                sys.exit(1)
            else:
                time_var = nc.variables['time']
                t = nc4.num2date(time_var[:], time_var.units)
                data = np.empty([len(t), 0])
                for i, n in enumerate(nodes):
                    z = nc['zeta'][:, n-1]
                    data = np.hstack((data, z))
                np.place(data, data < -1000, np.nan)
                df_sub = pd.DataFrame(data, columns=stationids, index=t)
                utilities.log.debug(" df_sub time range = {} -> {}" .format(df_sub.index[0], df_sub.index[-1]))
                df = pd.concat((df, df_sub), axis=0)
                utilities.log.debug("df_concat time range = {} -> {}" .format(df.index[0], df.index[-1]))
    utilities.log.info('Updating internal time range to reflect data values')
    timestart = df.index[0]
    timestop = df.index[-1]
    print('water_levels_63: ADC actual time range {} {}'.format(timestart, timestop))
    return df

def get_water_levels61(urls, stationids):
    """
    Retrieves ADCIRC water levels from nowcast URLS, using fort.61.nc files
    :param urls:
    :param stationids:
    :return: DataFrame of water ADCIRC water levels at station locations
    """
    df = pd.DataFrame(columns=stationids)

    # find stationid matches in fort61 station_name list
    firsturl = first_true(urls.values())
    ds = xr.open_dataset(firsturl)
    ds = ds.transpose()
    ds.attrs = []
    sn = ds['station_name'].values
    snn = []
    for i in range(len(sn)):
        ts = str(sn[i].strip().decode("utf-8"))
        snn.append(ts)
    print(snn)  # DEBUG
    idx = {}
    for ss in stationids:
        s = str(ss[0])
        if s in snn:
            # print("{} is in list.".format(s))
            idx[s] = snn.index(s)
        else:
            print("{} not in fort.61.nc station_name list".format(s))
            sys.exit(1)
    #print(idx.values())
    exit(0)
    for datecyc, url in urls.items():
        if url is None:
            print("Skipping {}. No url. " .format(datecyc))
            # assume this missing nowcast is 6 hours
            t = [datecyc + timedelta(hours=x) for x in range(6)]
            data = np.empty((len(t), len(nodes),))
            data[:] = np.nan
        else:
            print("Loading {}." .format(datecyc))
            nc = nc4.Dataset(url)
            if "zeta" not in nc.variables.keys():
                print("zeta not found in fort.61.nc for {}.".format(datecyc))
                sys.exit(1)
            else:
                time_var = nc.variables['time']
                t = nc4.num2date(time_var[:], time_var.units)
                data = np.empty([len(t), 0])
                for i, n in enumerate(nodes):
                    z = nc['zeta'][:, n-1]
                    data = np.hstack((data, z))
                np.place(data, data < -1000, np.nan)
        df_sub = pd.DataFrame(data, columns=stationids, index=t)
        df = pd.concat((df, df_sub), axis=0)
        # df.reset_index()
        # print(df.shape)
    return df

def first_true(iterable, default=False, pred=None):
    """
    itertools recipe found in the Python 3 docs
    Returns the first true value in the iterable.
    If no true value is found, returns *default*
    If *pred* is not None, returns the first item
    for which pred(item) is true.

    first_true([a,b,c], x) --> a or b or c or x
    first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    """
    return next(filter(pred, iterable), default)
    parser.add_argument('--timein', default="'2020-08-01 12:00'", help='Timein string', type=str)



if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('--ignore_pkl', help="Ignore existing pickle files.", action='store_true')
    parser.add_argument('--verbose', help="Ignore existing pickle files.", action='store_true')
    #parser.add_argument('--cdat', default=None, help='Date to run system for.', type=str)
    # parser.add_argument('--ccyc', default="-1", help='Cycle to run system for.', type=str)
    # parser.add_argument('--hoffset', default=0, help='Hour offset for WW3 download.', type=int)
    # parser.add_argument('--doffset', default=None, help='Day offset or datetime string for analysis', type=int)
    parser.add_argument('--doffset', default=None, help='Day lag or datetime string for analysis: def to YML -4', type=int)
    parser.add_argument('--timeout', default=None, help='YYYY-mm-dd HH:MM. Latest day of analysis def to now()', type=str)
    parser.add_argument('--timein', default=None, help='YYYY-mm-dd HH:MM.Starting day of analysis. def to dtime2 YTML value (-4)', type=str)
    parser.add_argument('--iometadata', action='store', dest='iometadata',default=None, help='Used to further annotate output files', type=str)
    parser.add_argument('--iosubdir', action='store', dest='iosubdir',default=None, help='Used to locate output files into subdir', type=str)
    # parser.add_argument('--noftp', action='store_true', help='ftp results ... ')
    # parser.add_argument('--nodropbox', action='store_true', help='ftp results ... ')
    # parser.add_argument('--experiment', default='Exp1', help='Experiment to analyze ... ')
    # parser.add_argument('--clobber', action='store_false', help='Clobber existing output files ... ')
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    parser.add_argument('--writeJson', action='store_true',help='Additionally stores resulting PKL data to a dict as a json')
    parser.add_argument('--variableName', action='store', dest='variableName', default=None,
                        help='String: Name used in DICT/JSon to identify ADCIRC type (eg nowcast,forecast)')
    parser.add_argument('--adcYamlname', action='store', dest='adcYamlname', default=None,
                        help='String: FQFN  of alternative config to adc.yml')

    args = parser.parse_args()

    sys.exit(main(args))

