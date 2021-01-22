
# Need large memory to run this job
import os
import sys
import pandas as pd
import json
import datetime
from utilities.utilities import utilities

print('Process the 52 separate reanalysis error files')

ERRDIR='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/WEEKLY/errorfield'
ADCJSON='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/adc_coord.json'
CLAMPFILE='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/config/clamp_list_hsofs.dat'
YMLNAME='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/config/int.REANALYSIS.yml'
ROOTDIR='/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools/pipelines/TEST3/WEEKLY'

# Set of all files belonging to this ensemble
errfileJson=ERRDIR+'/runProps.json'

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
    METADATA='_'+key  # Allows adjusting names of output files to include per-week
    print('Start week {}'.format(key))
    os.system('python krigListOfErrorSets.py  --errorfile '+ERRFILE+' --clampfile '+CLAMPFILE+' --gridjsonfile '+ADCJSON+' --iometadata '+ METADATA)

print('Completed ensemble')
