# Construct the data sets given the sizes n,p,k,and w
# Run the entire suite of data for our reanalysis project. PLan on direcvtories labled by year filled with files
# labled by month

# The default lon orientation is defEast, (0-360) so set the OWI converter parameters appropriately.

import time as tm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from shutil import copy2

#############################################################################
# Build a slurm file

def build_slurm(year,month):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 128:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 2')
    slurm.append('#SBATCH -J CDS'+year) 
    slurm.append('#SBATCH --mem-per-cpu 32000')
    slurm.append('echo "Begin the CDS query" ')
    slurm.append('export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/convert_to_owi"')
    slurm.append('python -u $dir/cds_request.py --year "'+year+'" --month "'+month+'"') 
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
yearlist = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]


yearlist = ['2018']
month = '01'

for iyear in yearlist:
    year = str(iyear)
    print('Processing CDS {}'.format(year))
    slurmFilename = build_slurm(year, month)
    cmd = 'sbatch ./'+slurmFilename
    print('Launching job {} as {}'.format(year,slurmFilename))
    os.system(cmd) 
