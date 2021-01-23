#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000

export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools:$PYTHONPATH
export RUNTIMEDIR=./YEARLY

# Build the yearly error file store in $RUNTIMEDIR
python yearlyReanalysis.py --urljson reanalysis.json

# Store files in $RUNTIMEDIR/WEEKLY/errorfield
export RUNTIMEDIR=./YEARLY
#python weeklyLowpassSampledError.py --inyear 2018 --yearlyDir $RUNTIMEDIR

# Interpolate
ERRDIR=$RUNTIMEDIR/errorfield
ADCJSON=$RUNTIMEDIR
CLAMPFILE=$PYTHONPATH/config/clamp_list_hsofs.dat
YAMLNAME=$RUNTIMEDIR/config/int.REANALYSIS.yml
ROOTDIR=$RUNTIMEDIR/WEEKLY

#python krigListOfErrorSets.py  --rootdir $ROOTDIR --yamlname $YAMLNAME --errorfile $ERRFILE --clampfile $CLAMPFILE --gridjsonfile $ADCIRCGRD
