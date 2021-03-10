#!/usr/bin/env python

# Need large memory to run this job
import os
import sys
import pandas as pd
import json
import datetime
from utilities.utilities import utilities

#############################################################################
# Build a slurm file

def build_slurm(ROOTDIR,YAMLNAME,ERRFILE,CLAMPFILE,ADCJSON,RANGE,SILL):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 24:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 1')
    slurm.append('#SBATCH -J Interpolate')
    slurm.append('#SBATCH --mem-per-cpu 64000')
    slurm.append('echo "Begin the Interpolation phase" ')
    slurm.append('export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/reanalysis"')
    slurm.append('python -u $dir/krigListOfErrorSets.py --daily --inrange "'+RANGE+'" --insill "'+SILL+'" --outroot "'+ROOTDIR+'" --yamlname "'+YAMLNAME+'" --errorfile "'+ERRFILE+'" --clampfile "'+CLAMPFILE+'" --gridjsonfile "'+ADCJSON+'"' )
    with open('runSlurm.sh', 'w') as file:
        for row in slurm:
            file.write(row+'\n')
    file.close()
    return ('runSlurm.sh')

def main(args):
    print('Process the separate reanalysis error files')
    utilities.log.info('Start the iterative interpolation pipeline')
    ERRDIR=args.errordir
    CLAMPFILE=args.clampfile
    ADCJSON=args.gridjsonfile
    YAMLNAME=args.yamlname
    ROOTDIR=args.outroot
    RANGE=str(args.inrange)
    SILL=str(args.insill)
    utilities.log.info('ERRDIR {}'.format(ERRDIR))
    utilities.log.info('CLAMPFILE {}'.format(CLAMPFILE))
    utilities.log.info('ADCJSON {}'.format(ADCJSON))
    utilities.log.info('YAMLNAME {}'.format(YAMLNAME))
    utilities.log.info('ROOTDIR {}'.format(ROOTDIR))
    utilities.log.info('RANGE {}'.format(RANGE))
    utilities.log.info('SILL {}'.format(SILL))

    # Set of all files belonging to this ensemble
    errfileJson=ERRDIR+'/runProps.json'
    with open(errfileJson, 'r') as fp:
        try:
            listedFiles = json.load(fp)
        except OSError:
            utilities.log.error("Could not open/read file {}".format(errfileJson))
            sys.exit()
    print('Begin the iteration')

    import random
    #for key, value in dict(random.sample(listedFiles.items(),20)).items(): # listedFiles.items():
    for key, value in listedFiles.items():
        print(key)
        print(value)
        ERRFILE=value
        print('Processing file {}'.format(ERRFILE))
        #METADATA='_'+key  # Allows adjusting names of output files to include per-week
        print('Start {}'.format(key))
        utilities.log.info('ERRFILE {}'.format(ERRFILE))
        utilities.log.info('CLAMPFILE {}'.format(CLAMPFILE))
        utilities.log.info('ADCJSON {}'.format(ADCJSON))
        utilities.log.info('YAMLNAME {}'.format(YAMLNAME))
        utilities.log.info('ROOTDIR {}'.format(ROOTDIR))
        utilities.log.info('RANGE {}'.format(RANGE))
        utilities.log.info('SILL {}'.format(SILL))
        slurmFilename = build_slurm(ROOTDIR,YAMLNAME,ERRFILE,CLAMPFILE,ADCJSON,RANGE,SILL)
        cmd = 'sbatch ./'+slurmFilename
        os.system(cmd)

        #if args.daily:
        #    os.system('python krigListOfErrorSets.py --daily  --outroot '+ROOTDIR+' --yamlname '+YAMLNAME+'  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON)
        #else:
        #    os.system('python krigListOfErrorSets.py --outroot '+ROOTDIR+' --yamlname '+YAMLNAME+'  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON)
    print('Completed ensemble')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('--errordir', action='store', dest='errordir',default=None, help='FQFN to stationSummaryAves_*.csv', type=str)
    parser.add_argument('--clampfile', action='store', dest='clampfile',default=None, help='FQFN to clamp_list_hsofs.dat', type=str)
    parser.add_argument('--gridjsonfile', action='store', dest='gridjsonfile',default=None, help='FQFN to ADCIRC lon,lat values. if = Skip then skip ADCIRC interpolation', type=str)
    parser.add_argument('--cv_kriging', action='store_true', dest='cv_kriging',
                        help='Boolean: Invoke a CV procedure prior to fitting kriging model')
    parser.add_argument('--yamlname', action='store', dest='yamlname', default=None)
    parser.add_argument('--daily', action='store_true', dest='daily',
                        help='Boolean: specify DAILY to the krig method')
    parser.add_argument('--inrange', action='store', dest='inrange',default=None, help='If specified then an internal config is constructed', type=int)
    parser.add_argument('--insill', action='store', dest='insill',default=None, help='If specified then an internal config is constructed', type=float)
    parser.add_argument('--outroot', action='store', dest='outroot', default=None,
                        help='Available high level output dir directory')
    args = parser.parse_args()
    sys.exit(main(args))
