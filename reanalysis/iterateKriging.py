
# Need large memory to run this job
import os
import sys
import pandas as pd
import json
import datetime
from utilities.utilities import utilities

print('Process the 52 separate reanalysis error files')

#ERRDIR='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/WEEKLY/errorfield'
#ADCJSON='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/adc_coord.json'
#CLAMPFILE='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/config/clamp_list_hsofs.dat'
#YMLNAME='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/config/int.REANALYSIS.yml'
#ROOTDIR='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/WEEKLY'

#ERRDIR=$RUNTIMEDIR/errorfield
#ADCJSON=$RUNTIMEDIR
#CLAMPFILE=$PYTHONPATH/config/clamp_list_hsofs.dat
#YMLNAME=$RUNTIMEDIR/config/int.REANALYSIS.yml
#ROOTDIR=$RUNTIMEDIR/WEEKLY

# Set of all files belonging to this ensemble
errfileJson=ERRDIR+'/runProps.json'

def main(args):

    ERRDIR=args.errordir
    CLAMPDIR=args.clampdir
    ADCJSON=args.gridjsonfile
    YAMLNAME=args.yamlname
    ROOTDIR=args.rootdir
    utilities.log.info('ERRDIR {}'.format(ERRDIR))
    utilities.log.info('CLAMPDIR {}'.format(CLAMPDIR))
    utilities.log.info('ADCJSON {}'.format(ADCJSON))
    utilities.log.info('YAMLNAME {}'.format(YSMLNAME))
    utilities.log.info('ROOTDIR {}'.format(ROOTDIR))

    with open(errfileJson, 'r') as fp:
        try:
            weeklyFiles = json.load(fp)
        except OSError:
            utilities.log.error("Could not open/read file {}".format(errfileJson))
            sys.exit()
    print('Begin the iteration')
    for key, value in weeklyFiles.items():
        print(key)
        print(value)
        ERRFILE=value
        #METADATA='_'+key  # Allows adjusting names of output files to include per-week
        print('Start week {}'.format(key))
        os.system('python krigListOfErrorSets.py  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON)
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
    parser.add_argument('--rootdir', action='store', dest='rootdir', default=None,
                        help='Available high level output dir directory')

