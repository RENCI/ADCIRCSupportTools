#Example outpuit files one is pressure and one is wind velocity
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.pre
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.win

# BEWARE of how slice works on Xarray. Sorting is critical for proper function and
# errors cam be hard to find
#
# The user CLIU is a little cumbersom. On input to specify the location of the ERA5
# monthly files, we pass in the actual full path and this code will CONSTRUCT
# input filenames as MM.nc
# For the output OWI files we pass in the BASEDIR underwhich files will be stored.
#
#
"""era5_to_owi
    Simple code to read an ERA5 file and dump out some statistics about it.
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
from convert_to_owi import era5_to_owi

###################################################
# Begin
# Build a list of FQ filenames for saving to a file and passing to the converter.
#
date_start='201812'  # YYYYMM
date_end='201901'  # YYYYMM
ddir='/projects/ees/TDS/ERA5/global'

def assembleList(ddir,date_start,date_end):
    """
    Simple creates a list of FQFN for passing to the converter
    We expect these to be ERA5 files predownloaded
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

###################################################
# Begin
#[build list of files]

def main(args):
    print('This pipeline assumes netCDF data filenames in the format MM.nc. Where MM is the month')

    date_start = args.date_start # 'YYYYMM'
    date_end = args.date_end
    metaname = args.metaname # For example '2018' if merging 2018+- 1 mnth flank

    ddir = args.ddir # for example /projects/ees/TDS/ERA5/global

    odir = args.odir # for example /projects/ees/TDS/Reanalysis/Winds

    mergefiles = args.merge
    convertEastWest = args.convertEastWest

    config = utilities.load_config() # Defaults to main.yml as specified in the config

    #Rootdir not used in this example

    if odir==None:
        odir=utilities.fetchBasedir(config['DEFAULT']['RDIR'], basedirExtra='Reanalysis/Winds')

    utilities.log.info('Specified METANAME to process {}'.format(metaname))
    utilities.log.info('Directory to find data {}'.format(ddir))
    utilities.log.info('Directory to write data {}'.format(odir))
    utilities.log.info('Merge files requested {}'.format(mergefiles))
    utilities.log.info('Must we transform data to EastWest convention ?{}'.format(convertEastWest))

    ##We can still keep runTIMEDIR for temp files like merged.nc
    ##dir='/home/jtilson/ADCIRCSupportTools/convert_to_owi'
    ##dir='/projects/sequence_analysis/vol1/prediction_work/ERAtoOWI/ERA5'
    
    filelist,monthlist = assembleList(ddir,date_start,date_end)
    utilities.log.info('Files to process {}'.format(filelist))
    
    # Get fetch/boundary characteristics
    owi_yaml=os.path.join('/home/jtilson/ADCIRCSupportTools', 'config', 'owi.yml')
    owi_config = utilities.load_config(owi_yaml)
    parameters = ('msl','u10','v10') # Need to include a check for their existence
    inboxlons = era5_to_owi.loadBoundaries(owi_config,'lon')
    inboxlats = era5_to_owi.loadBoundaries(owi_config,'lat')
    inboxtimes = era5_to_owi.loadBoundaries(owi_config,'time') # None means use all available times
    utilities.log.info('Input lons {}'.format(inboxlons))
    utilities.log.info('Input lats {}'.format(inboxlats))
    utilities.log.info('Input times {}'.format(inboxtimes))
    
    measurements = owi_config['OWI']['measurements']

    subdir=metaname # SHould be a year but could be anything
    
    if mergefiles:
        t0 = tm.time()
        fname='mergedData.nc'
        ds_merged = era5_to_owi.mergeDatasets(filelist)
        ofilename=utilities.getSubdirectoryFileName(odir,subdir,fname)
        metaData = 'merged'
        era5_to_owi.saveMergedDataset(ds_merged,ofilename)
        filelist=[ofilename]
        utilities.log.info('Save MERGED file {}. Time was {}'.format(ofilename,tm.time()-t0))
        months = [metadata]
    iometadata=''
    
    t0 = tm.time() 
    for era5file,month in zip(filelist,months):
        print(era5file)
        ds = era5_to_owi.transformCoordinates(xr.open_dataset(era5file), convertEastWest )
        inboxlats =  era5_to_owi.conformBoundaries(ds.latitude, inboxlats)
        inboxlons =  era5_to_owi.conformBoundaries(ds.longitude, inboxlons)
        inboxtimes = era5_to_owi.conformBoundaries(ds.time, inboxtimes, keyname='time')
        # Now potentially subselect data
        ds_subset = era5_to_owi.fetchSubArray(ds, inboxlats, inboxlons, inboxtimes)
        for measurement in measurements:   # pressure of velocity for now
            utilities.log.info('Processing file {} and measurement {}'.format(era5file,measurement))
            fname = era5_to_owi.globalFilenames[measurement].replace('fort',month)
            filename = utilities.getSubdirectoryFileName(odir, subdir, fname)
            era5_to_owi.process_filedata(ds_subset, filename, measurement)
        utilities.log.info('Saved all files. Time was {}'.format(tm.time()-t0))
    utilities.log.info('Finished')

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    #parser.add_argument('--merge', default=False, dest=merge, help='merge dta into single file', type=str2bool)
    parser.add_argument('--ddir', type=str, action='store', dest='ddir', 
                        default='/projects/sequence_analysis/vol1/prediction_work/ADCIRC-ASSIMILATION-PROTOTYPES/ObservedWaterLevels/ERA5/2018',
                        help='Location of one or more ERA5 monthly netCDF files')
    parser.add_argument('--odir', type=str, action='store', dest='odir',
                        default=None, help='Location to write one or more OWI monthly files')
    parser.add_argument('--merge', type=str2bool, action='store', dest='merge', default=True,
                        help='Boolean: Choose to combine all files into a single OWI')
    parser.add_argument('--convertEastWest', type=str2bool, action='store', dest='convertEastWest', default=True,
                        help='Boolean: Triggeer to True to force data to be transformed to EastWest coordinates')
    parser.add_argument('--metaname', type=str, action='store', dest='metaname', default='',
                        help='Should be Year (YYYY) to for (merged) output file')
    parser.add_argument('--date_start', type=str, action='store', dest='date_start',
                        default=None, help='YYYYMM')
    parser.add_argument('--date_end', type=str, action='store', dest='date_end',
                        default=None, help='YYYYMM')
    args = parser.parse_args()
    sys.exit(main(args))



