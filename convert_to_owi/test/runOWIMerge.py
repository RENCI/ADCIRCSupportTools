# COnvert a years work of ERA5 data to OWI. Pad data with with a 1 month flank 
#

import time as tm
import pandas as pd
import numpy as np
import os, sys

#############################################################################
# Build a slurm file

# --ddir --odir --convertEastWest --metaname --date_start --date_end
def build_slurm(ddir,odir,date_start,date_end,year,extra):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 128:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 2')
    slurm.append('#SBATCH -J OWI'+year) 
    slurm.append('#SBATCH --mem-per-cpu 64000')
    slurm.append('echo "Begin the OWI processing" ')
    slurm.append('export PYTHONPATH=/home/jtilson/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/home/jtilson/ADCIRCSupportTools/convert_to_owi"')
    slurm.append('python -u $dir/era5_to_owi.py --metafilename "'+year+'" --metadirname "'+year+'" --convertEastWest --ddir "'+ddir+'" --date_start "'+date_start+'" --date_end "'+date_end+'"')
    shName = '_'.join([year,'runSlurm.sh'])
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

# For merging
#yearlist = [1980,1981,1982,1983,1984,1985,1986,1987,1988,1989]
#yearlist = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
#yearlist = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]
#yearlist = [2010,2011,2012,2013,2014,2015,2016,2017,2018]

yearlist = [2018]

ncBaseDir='/projects/ees/TDS/ERA5/global'
owiBasedir=None # Will pick up from RUNTIMEDIR

#
#>>> for d in [2018,2019,2020]:
# date_start print('%4d12' %(d-1))
# date_end print('%4d01' %(d+1))

# Merge all of 2018 and add 1 month flank
for iyear in yearlist:
    #date_start='201712'
    #date_end='201801'
    date_start = '%4d12' %(iyear-1)
    date_end ='%4d01' %(iyear+1)
    year = str(iyear)
    print('Processing CDS {}'.format(year))
    ddir = ncBaseDir
    odir = owiBasedir # Create this directory insoide of era5
    print('Processing DDIR {}'.format(ddir))
    #slurmFilename = build_slurm(ddir,odir,months,year)
    slurmFilename = build_slurm(ddir,odir,date_start,date_end,year,year)
    cmd = 'sbatch ./'+slurmFilename
    print('Launching job {} as {} '.format(ddir,odir))
    os.system(cmd) 
print('Finished')

