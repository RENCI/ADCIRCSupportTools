#!/usr/bin/env python

#Example outpuit files one is pressure and one is wind velocity
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.pre
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.win

# BEWARE of how slice works on Xarray. Sorting is critical for proper function and
# errors cam be hard to find
#
#
#
"""era5_to_owi
   This method wil produce ONE fort.221/fort.222 file pair. If the input start,end dates
   are the same, then a simple monthly OWI is created. If a range of dates are inputted, then
   as merged file is created. THis is a change in the user API from the prior version. The normal 
   range checks, etc are maintained as they should be consistent accross multiple dates if specified.
"""
import os,sys
import datetime as dt  
import numpy as np
import xarray as xr
import pandas as pd
import time as tm
from argparse import ArgumentParser
from utilities.utilities import utilities,str2bool
from requests.exceptions import ConnectionError,Timeout,HTTPError

#reset these to properties
globalBoundaries={'lon':[-99.00,-55.00], 'lat':[5.0, 50.0],'time': None}
globalFilenames={'pressure':'fort.221', 'velocity':'fort.222'}
globalDataTypes={'pressure':['msl'], 'velocity': ['u10','v10']}

###################################################

#TODO deal with the silly timestamp bungling that must go on. Especially the 
# specification of nanosecs. Who knows how often that will change
def checkRange(ds_coord, boxin=None, keyname=None):
    """
    See if the boxins range makes sense and adjust as necc
    if None, keep entire range
    Works for lat,lon,time. But lpon range must be the same style as data
    Also potentially sorts into min,max order
    We want to be able to subset data on the full ds data set
    Need to treat times differently. Needlessly cumbersome because time behaves differently in this context
    The application of this to TIMES is fraught will problems. Better to use all times
    """
    if boxin==None:
        print('None')
        box = [ds_coord.min().item(0), ds_coord.max().item(0)]
        if keyname=='time':
            box = [np.datetime64(t,'ns') for t in box]
        print('Empty boxin. Taking full range {},{}'.format(*box))
    else:
        boxmax = max(boxin) if np.logical_and(max(boxin)<=ds_coord.max(),max(boxin)>ds_coord.min()) else ds_coord.max().values
        boxmin = min(boxin) if np.logical_and(min(boxin)>=ds_coord.min(),min(boxin)<boxmax) else ds_coord.min().values
        box = [boxmin, boxmax]
    print('Input BOX range retained as {},{}'.format(*box))
    return box

##################################################################################
# Most of this IO code grabbed/adapted from B.O.Blanton work

time_string_template='iLat=%4diLong=%4dDX=%6.4fDY=%6.4fSWLat=%8.5fSWLon=%8.4fDT=%12s'

def openowifile(filename):
    f = open(filename, "w")
    return f 

def writeowiheader(ofile,ds):
    dt=pd.to_datetime(str(ds['time'].values[0])).strftime('%Y%m%d%H%M')
    tstart = pd.to_datetime(str(ds['time'].values[0])).strftime('%Y%m%d%H')
    tend = pd.to_datetime(str(ds['time'].values[-1])).strftime('%Y%m%d%H')
    header="Oceanweather WIN/PRE Format                        %12s     %12s" % (tstart,tend)
    ofile.write('{}\n'.format(header))

def writeowiheadermanual(ofile,times):
    tstart,tend = times[0],times[1]
    header="Oceanweather WIN/PRE Format                        %12s     %12s" % (tstart,tend)
    ofile.write('{}\n'.format(header))

def closeowifiles(ofile):
    ofile.close()

# Need to loop inside here for msl versus u10/v10
# globalDataTypes={'pressure':'msl', 'velocity': ['u10','v10']}
def writesnap(ofile,datatype,ds):
    nlo=ds.dims['longitude']
    nla=ds.dims['latitude']
    swlat=ds['latitude'].values[0]
    swlon=ds['longitude'].values[0]
    dx=ds['longitude'].values[1]-swlon # Does this work for degEast orientation?
    dy=ds['latitude'].values[1]-swlat
    dt=pd.to_datetime(str(ds['time'].values)).strftime('%Y%m%d%H%M')
    timeline=time_string_template%(nla,nlo,dx,dy,swlat,swlon,dt)
    ofile.write(timeline+"\n") 
    for obs in globalDataTypes[datatype]:
        data = ds[obs].values.flatten()
        im=int(np.ceil(nlo*nla/8))
        for i in range(im):
            temp=data[i*8:(i+1)*8].tolist()
            temp = ["%9.4f" % k for k in temp]
            s=' '.join(temp)
            ofile.write(" "+s+"\n")

def process_filedata(ds, ofile, datatype):
    """
    Process a single measurement type: currently only pressure (msl) or pressure (v10,u10)
    Loops over times writing out lat/lons blocks
    """
    utilities.log.info('Generate SNAP for measurement {}'.format(datatype))
    nt = ds.dims['time']
    for i in range(nt):
        writesnap(ofile,datatype,ds.isel(time=i))

def loadBoundaries(config, keyname):
    """
    Read config for the keyname and set to the output. If key doesnt
    exist choose the default from the global dict
    If keyname is time ensure the resuilts are datetime64 objects
    """
    try:
        if keyname=='time':
            inbox = [np.datetime64(t) for t in config['OWI'][keyname]]
        else:
            inbox = config['OWI'][keyname]
    except KeyError:
        inbox=globalBoundaries[keyname]
        utilities.log.info('Boundary box not found in config: use default {}'.format(inbox))
    return inbox

def conformBoundaries(ds_coord, inbox, keyname=None):
    box = checkRange(ds_coord, inbox, keyname)
    utilities.log.info('Final ranges: {},{}'.format(*box))
    return box

#TODO check on the sorting issue, errors here can be very hard to find
def transformCoordinates(ds, convertEastWest=False):
    """
    Transform lon,lat, and pressure magnitudes
    """
    t0=tm.time()
    if convertEastWest:
        # Now we want to modify the longitude coords to be east/west style.
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds=ds.reindex(latitude=ds.latitude[::-1]) # lats are north-to-south. Flip, we need south-to-north
    ds=ds.sortby('longitude','latitude') # Need this 
    ds['msl']=ds['msl']/100 # convert pres from Pa to mb\n"
    utilities.log.info('Current Xarray has been transformed to desired coordinates and units')
    utilities.log.info('Time to transform coors is {}'.format(tm.time()-t0))
    return ds
 
def fetchSubArray(ds, inlats, inlons, intimes):
    """
    Collect the coimplicated slicing comand into this method. Returns a new Xarray
    On the boundaries inclusive
    NOTE: some coordinate dependancies exists. For example for lons slice must go from lowest to highest.
    It would be better to convert everything to DegEast 
    """
    ds_subset = ds.sel({'latitude': slice(inlats[0],inlats[1]),
            'longitude': slice(inlons[0],inlons[1]), #Must be orderd from neg to positive
            'time':slice(intimes[0],intimes[1])})
    return ds_subset

def mergeDatasetsBigMemory(files):
    """
    Merge the Xarray data for the provided files into a single Xarray object
    No filtering or coordinate transformations are applied
    Likely to be a large memory operation
    """
    dataX = list()
    for f in files:
        dataX.append(xr.open_dataset(f))
    utilities.log.info('Merging {} data sets'.format(len(dataX)))
    xr_merged = xr.merge(dataX)
    return xr_merged

def mergeDatasets(files):
    """
    Merge the Xarray data for the provided files into a single Xarray object
    No filtering or coordinate transformations are applied
    Less efficient but also less memory
    """
    init = True
    for f in files:
        nc = xr.open_dataset(f)
        if init==True:
            xr_merged = nc
            init=False
        else:
            xr_merged = xr.merge([xr_merged,nc])
    utilities.log.info('Reduced memory Merging {} data sets'.format(len(files)))
    #xr_merged = xr.merge(dataX)
    return xr_merged

def saveMergedDataset(ds_merged,filename):
    """
    If creating a merged data set save for posterity
    """
    ds_merged.to_netcdf(filename)
    utilities.log.info('Saved merged data set to disk {}'.format(filename))

def assembleList(ddir,date_start,date_end):
    """
    Creates a list of input FQFN for passing to the converter
    We expect these to be ERA5 files predownloaded
    For now assume local files
    A single month is possible
    """
    import datetime as dt
    import datedelta
    #urlpat="http://tds.renci.org:8080/thredds/dodsC/MultiDecadal-ERA5/global/%4d/%02d.nc"
    #url="http://tds.renci.org:8080/thredds/fileServer/MultiDecadal-ADCIRC/Winds/%4d/%02d.221"
    pathfile='/'.join([ddir,'%4d','%02d'])
    date_start=dt.datetime.strptime(date_start, '%Y%m')
    date_end=dt.datetime.strptime(date_end, '%Y%m')
    date_start + datedelta.MONTH
    now=date_start
    flistnc = list()
    months=list()
    while now <= date_end: # in monthly strides
        fnc = '.'.join([pathfile % (now.year,now.month),'nc'])
        flistnc.append(fnc)
        months.append('%02d' % (now.month))
        now=now+datedelta.MONTH
    return flistnc,months

# What about time range? pd.to_datetime(str(ds['time'].values[0])).strftime('%Y%m%d%H')
# Why do numpy datetime64, pandas datetime have to behave differently?

def build_combined_header(filelist, monthlist, ddir, inboxtimes=None):
    """
    Get first and last entry, read header and parse times
    """
    fstart = filelist[0]
    fend = filelist[-1]
    if inboxtimes != None:
        utilities.log.error('inboxtimes must be None')
    ds = xr.open_dataset(fstart)
    starttime = ds['time'].min().item(0)
    tstart = pd.Timestamp(starttime,'ns').strftime('%Y%m%d%H')
    #
    ds = xr.open_dataset(fend)
    endtime = ds['time'].max().item(0)
    tend = pd.Timestamp(endtime,'ns').strftime('%Y%m%d%H')
    header="Oceanweather WIN/PRE Format                        %12s     %12s" % (tstart,tend)
    utilities.log.info('File header determined to be {}'.format(header))
    return header

##########################################################
## Begin
#[build list of files]

def main(args):
    print('This pipeline assumes netCDF data filenames in the format MM.nc. Where MM is the month')

    ddir = args.ddir # for example /projects/ees/TDS/ERA5/global
    odir = args.odir # for example /projects/ees/TDS/Reanalysis/Winds
    convertEastWest = args.convertEastWest
    metadirname = args.metadirname
    metafilename = args.metafilename
    date_start = args.date_start
    date_end = args.date_end   # inclusive
    basedirExtra = args.basedirextra
    config = utilities.load_config() # Defaults to main.yml as specified in the config
    #Rootdir not used in this example
    if odir==None:
        #odir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='Reanalysis/Winds')
        odir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra=basedirExtra)

    utilities.log.info('Specified METADIRNAME METAFILENAME to process {}, {}'.format(metadirname,metafilename))
    utilities.log.info('Directory to find data {}'.format(ddir))
    utilities.log.info('Directory to write data {}'.format(odir))
    utilities.log.info('Date start {}, Date end {}'.format(date_start, date_end))
    utilities.log.info('Must we transform data to EastWest convention ?{}'.format(convertEastWest))

    filelist,monthlist = assembleList(ddir,date_start,date_end)
    utilities.log.info('ERA5 Files to process {}'.format(filelist))

    # Get desired OUTPUT characteristics from the OWI YAML
    owi_yaml = os.path.join(os.path.dirname(__file__), "../config/", "owi.yml")
    owi_config = utilities.load_config(owi_yaml)

    parameters = ('msl','u10','v10') # Need to include a check for their existence

    inboxlons = loadBoundaries(owi_config,'lon')
    inboxlats = loadBoundaries(owi_config,'lat')
    inboxtimes = loadBoundaries(owi_config,'time') # None means use all available times
    utilities.log.info('Input lons {}'.format(inboxlons))
    utilities.log.info('Input lats {}'.format(inboxlats))
    utilities.log.info('Input times {}'.format(inboxtimes))

    measurements = owi_config['OWI']['measurements']

    # Specify OUTPUT filenames include metaname (aka year) attribute

    outfiles={}
    for measurement in measurements:
        outfiles[measurement]=utilities.getSubdirectoryFileName(odir,metadirname,globalFilenames[measurement].replace('fort',metafilename))
    utilities.log.info('ofiles are: {}'.format(outfiles))

    # Begin the work
    fileheader = build_combined_header(filelist, monthlist, ddir, inboxtimes=None)

    # OPen the files 
# utilities.getSubdirectoryFileName(odir, metaname, fname)
    ofiles={}
    for measurement in measurements:
        ofiles[measurement] = openowifile(outfiles[measurement]) 

    # Write the headers
    for key, of in ofiles.items():
        of.write('{}\n'.format(fileheader)) # Same header for both .221 and .222

    #Open output files and write headers
    t0 = tm.time() 

    for era5file,month in zip(filelist,monthlist):
        utilities.log.debug('Processing file {}, month{}'.format(era5file,month))
        ds = transformCoordinates(xr.open_dataset(era5file), convertEastWest )
        inboxlats =  conformBoundaries(ds.latitude, inboxlats)
        inboxlons =  conformBoundaries(ds.longitude, inboxlons)
        inboxtimes = conformBoundaries(ds.time, inboxtimes, keyname='time')
        # Now subselect data: This will nearly always happen. Not many check for consistency yet
        ds_subset = fetchSubArray(ds, inboxlats, inboxlons, inboxtimes)
        for measurement in measurements:   # pressure of velocity for now
            utilities.log.info('Processing file {} and measurement {}'.format(era5file,measurement))
            process_filedata(ds_subset, ofiles[measurement], measurement)
        utilities.log.info('Saved all files. Time was {}'.format(tm.time()-t0))
    utilities.log.info('Finished')

    # Close files
    for key, of in ofiles.items():
        of.close()
    utilities.log.info('Saved and closed all files. Time was {}'.format(tm.time()-t0))

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    #parser.add_argument('--merge', default=False, dest=merge, help='merge dta into single file', type=str2bool)

    parser.add_argument('--ddir', type=str, action='store', dest='ddir',
                        default='/projects/ees/TDS/ERA5/global',
                        help='Location of one or more ERA5 monthly netCDF files')
    parser.add_argument('--odir', type=str, action='store', dest='odir',
                        default=None, help='Location to write one or more OWI monthly files')
    parser.add_argument('--convertEastWest', action='store_true',
                        help='Boolean: Triggeer to True to force data to be transformed to EastWest coordinates')
    parser.add_argument('--basedirextra', type=str, action='store', dest='basedirextra', default='Reanalysis/Winds',
                        help='Appends to $RUNTIMEDIR if odir not specified')
    parser.add_argument('--metadirname', type=str, action='store', dest='metadirname', default='2018',
                        help='Should be Year (YYYY) to for (merged) output file')
    parser.add_argument('--metafilename', type=str, action='store', dest='metafilename', default='2018',
                        help='Should be Info for output file. If monthly set to something like 01')
    parser.add_argument('--date_start', type=str, action='store', dest='date_start',
                        default='201712', help='YYYYMM')
    parser.add_argument('--date_end', type=str, action='store', dest='date_end',
                        default='201901', help='YYYYMM')
    args = parser.parse_args()
    sys.exit(main(args))



