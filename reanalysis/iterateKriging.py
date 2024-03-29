#!/usr/bin/env python

# Need large memory to run this job
import os
import sys
import pandas as pd
import json
import datetime
from utilities.utilities import utilities

def main(args):
    print('Process the separate reanalysis error files')
    utilities.log.info('Start the iterative interpolation pipeline')
    ERRDIR=args.errordir
    CLAMPFILE=args.clampfile
    ADCJSON=args.gridjsonfile
    YAMLNAME=args.yamlname
    ROOTDIR=args.outroot
    utilities.log.info('ERRDIR {}'.format(ERRDIR))
    utilities.log.info('CLAMPFILE {}'.format(CLAMPFILE))
    utilities.log.info('ADCJSON {}'.format(ADCJSON))
    utilities.log.info('YAMLNAME {}'.format(YAMLNAME))
    utilities.log.info('ROOTDIR {}'.format(ROOTDIR))

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
        addstring=' '
        if args.inrange is not None:
            addstring='--inrange '+str(args.inrange)
        if args.insill is not None:
            addstringSill='--insill '+str(args.insill)
        if args.daily:
            os.system('python krigListOfErrorSets.py --daily '+addstring+' '+addstringSill+'  --outroot '+ROOTDIR+' --yamlname '+YAMLNAME+'  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON)
        else:
            os.system('python krigListOfErrorSets.py '+addstring+' --outroot '+ROOTDIR+' --yamlname '+YAMLNAME+'  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON)
        sys.exit('Manual exit')

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
    parser.add_argument('--outroot', action='store', dest='outroot', default=None,
                        help='Available high level output dir directory')
    parser.add_argument('--inrange', action='store', dest='inrange',default=None, help='If specified then an internal config is constructed', type=int)
    parser.add_argument('--insill', action='store', dest='insill',default=None, help='If specified then an internal config is constructed', type=float)
    args = parser.parse_args()
    sys.exit(main(args))
