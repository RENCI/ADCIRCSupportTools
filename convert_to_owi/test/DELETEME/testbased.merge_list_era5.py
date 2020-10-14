#Example outpuit files one is pressure and one is wind velocity
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.pre
#/projects/storm_surge/COMT/FlorenceOWI/2018_06_NOATL_28Km_Basin.win

"""era5_to_owi
    Simple code to read an ERA5 file and dump out some statistics about it.
    Merge IN ORDER several fort 22{1/2} files int a single file with flanking.
    No checks are performed on file validity. Build new header from the 
    first and last provided file times.
"""
import os,sys
import datetime as dt  
import numpy as np
import pandas as pd
import time as tm
from argparse import ArgumentParser
from utilities.utilities import utilities
#from convert_to_owi import era5_to_owi

###################################################
# Begin
# Build a list of FQ filenames for saving to a file and passing to the converter.
#
date_start='201812'  # YYYYMM
date_end='201901'  # YYYYMM
ddir='/projects/ees/TDS/ERA5/global'

def assembleList(ddir,date_start,date_end, measurement='221'):
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
    now=date_start
    flistnc = list()
    months=list()
    while now <= date_end: # in monthly strides
        fnc = '.'.join([pathfile % (now.year,now.month),measurment])
        flistnc.append(fnc)
        months.append('%02d' % (now.month))
        now=now+datedelta.MONTH
    return flistnc,months

def build_combined_header(filelist, monthlist, ddir):
    """
    Get first and last entry, read header and parse times
    """
    fstart = filelist[0]
    fend = filelist[-1]
    reader = open(fstart)
    headers=list()
    for f in[fstart,fend]:
        try:
            header.append(reader.readline())
        except FileNotFoundError:
            utilities.log.error('File not found {}'.format(fstart))
        reader.close()
    tstart=header[0].split()[3]
    tend=header[1].split()[4]
    header="Oceanweather WIN/PRE Format                        %12s     %12s" % (tstart,tend)
    return header

###################################################
# Begin
#[build list of files]

def main(args):
    print('This pipeline assumes OWI data filenames are in the format MM.221 and MM.222. Where MM is the month')

    # Read in strings of the form YYYYMM indicating the start and end files
    # Read in metadata that wil be used to 1) build the list and name the output file. Eg, 
    # metadata ==2018 wil create  files 2018.221/2018.222

    date_start = args.date_start # 'YYYYMM'
    date_end = args.date_end
    metaname = args.metaname # For example '2018' if merging 2018+- 1 mnth flank

    ddir = args.ddir # for example /projects/ees/TDS/ERA5/global
    odir = args.odir # for example /projects/ees/TDS/Reanalysis/Winds

    utilities.log.info('Specified METANAME to process {}'.format(metaname))
    utilities.log.info('Directory to find data {}'.format(ddir))
    utilities.log.info('Directory to write data {}'.format(odir))
    utilities.log.info('Dates: Start {}, end {}'.format(date_start,date_end))

    measurement='221'
    filelist,monthlist = assembleList(ddir,date_start,date_end,measurement)
    utilities.log.info('Files to process {}'.format(filelist))
    
    combined_header = build_combined_header(filelist, monthlist, ddir)

    owifile='junk.221' 
    ofile = open(owifile,'w')
    ofile.write('{}\n'.format(combined_header))
    for owifile,month in zip(filelist,monthlist):
        print(owifile)
        f = open(owifile,'r')
        lines = f.readlines()[1:]
        f.close()
        #
        ofile.writelines(lines)

    ofile.close()
    utilities.log.info('Finished')

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    #parser.add_argument('--merge', default=False, dest=merge, help='merge dta into single file', type=str2bool)
    parser.add_argument('--ddir', type=str, action='store', dest='ddir', 
                        default='/projects/sequence_analysis/vol1/prediction_work/ADCIRC-ASSIMILATION-PROTOTYPES/ObservedWaterLevels/ERA5/2018',
                        help='Location of one or more OWI monthly netCDF files')
    parser.add_argument('--odir', type=str, action='store', dest='odir',
                        default=None, help='Location to write one or more OWI monthly files')
    parser.add_argument('--metaname', type=str, action='store', dest='metaname', default='',
                        help='Should be Year (YYYY) to for (merged) output file')
    parser.add_argument('--date_start', type=str, action='store', dest='date_start',
                        default=None, help='YYYYMM')
    parser.add_argument('--date_end', type=str, action='store', dest='date_end',
                        default=None, help='YYYYMM')
    args = parser.parse_args()
    sys.exit(main(args))



