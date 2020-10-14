# Convert each months worth of CDS ERA5 data to OWI. 
# Note the --convertEastWest True correcvtion for the input degEast lon orientation
#
# This code can MERGE months together as well.
# See runMerge.sh
#

import time as tm
import pandas as pd
import numpy as np
import os, sys

#############################################################################
# Build a slurm file

# --ddir --odir --convertEastWest --metaname --date_start --date_end
def build_slurm(ddir,odir,date_start,date_end,year,month):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 128:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 2')
    slurm.append('#SBATCH -J OWI'+year) 
    slurm.append('#SBATCH --mem-per-cpu 64000')
    slurm.append('echo "Begin the OWI processing" ')
    slurm.append('export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/convert_to_owi"')
    slurm.append('python -u $dir/era5_to_owi.py --metafilename "'+month+'" --metadirname "'+year+'" --convertEastWest --ddir "'+ddir+'" --date_start "'+date_start+'" --date_end "'+date_end+'"')
    shName = '_'.join([month,year,'runSlurm.sh'])
    with open(shName, 'w') as file:
        for row in slurm:
            file.write(row+'\n')
    file.close()
    return (shName)

#################################################################################
# Setup a subdir based on the basename
# Why does Slurm in 2019 still not support the $1 CLI conventions?

#yearlist = [1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989]
#yearlist = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
#yearlist = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]
#yearlist = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

yearlist = [2018]

#ncBaseDir = '/projects/ees/TDS/ERA5/global'
#owiBasedir = '/projects/sequence_analysis/vol1/prediction_work/junk'

ncBaseDir='/projects/ees/TDS/ERA5/global'
owiBasedir=None # Will pick up from RUNTIMEDIR

#months=['01','02','03','04','05','06','07','08','09','10','11','12']
months=['01']

for iyear in yearlist:
    for month in months:
        date='%4d' % (iyear) + month
        date_start = date
        date_end = date
        year = str(iyear)
        print('Processing CDS {}'.format(year))
        ddir = ncBaseDir
        odir = owiBasedir # Create this directory insoide of era5
        print('Processing DDIR {}'.format(ddir))
        slurmFilename = build_slurm(ddir,odir,date_start,date_end,year,month)
        cmd = 'sbatch ./'+slurmFilename
        print('Launching job {} as {} '.format(ddir,odir))
print('Finished')

