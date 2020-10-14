# Construct the data sets given the sizes n,p,k,and w
#
# Current run. Best iterative scheme. We cell scale and log2(c+1) the input R once. Then we iterativelly solve the imputation.
# The regresssion R is unchanged and S is feature scaled only
# At the conclusion R is descaled

import time as tm
import pandas as pd
import numpy as np
import os, sys

#############################################################################
# Build a slurm file

def build_slurm(ddir, odir, smonths, year):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 128:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 2')
    slurm.append('#SBATCH -J OWI'+year) 
    slurm.append('#SBATCH --mem-per-cpu 32000')
    slurm.append('echo "Begin the OWI processing" ')
    slurm.append('export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/home/jtilson/ADCIRCSupportTools/convert_to_owi"')
    #slurm.append('python -u $dir/era5_to_owi.py --convertEastWest True --ddir "'+ddir+'" --merge False --months '+'"'+smonths+'"')
    slurm.append('python -u $dir/era5_to_owi.py --year 2018 --convertEastWest True --ddir "'+ddir+'" --odir "'+odir+'" --merge False --months '+'"'+smonths+'"')
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

yearlist = [2018]
ncBaseDir = '/projects/ees/TDS/ERA5/global'
owiBasedir = '/projects/sequence_analysis/vol1/prediction_work/junk'

ncBaseDir='/projects/sequence_analysis/vol1/prediction_work/VALIDATE_ERA5/GLOBAL'
owiBasedir='/projects/sequence_analysis/vol1/prediction_work/VALIDATE_ERA5/GLOBAL/OWI'



#months='01,02,03,04,05,06,07,08,09,10,11,12'
months='01'

for iyear in yearlist:
    year = str(iyear)
    print('Processing CDS {}'.format(year))
    ddir = ncBaseDir
    odir = owiBasedir # Create this directory insoide of era5
    print('Processing DDIR {}'.format(ddir))
    slurmFilename = build_slurm(ddir,odir,months,year)
    cmd = 'sbatch ./'+slurmFilename
    print('Launching job {} as {}'.format(ddir,odir,months))
    #os.system(cmd) 

