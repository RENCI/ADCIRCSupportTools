#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000

export YEAR=2018


export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools 
export PYTHONPATH=$CODEBASE:$PYTHONPATH
export RUNTIMEDIR=.
export BASEDIREXTRA=TESTFULL/STATE/YEARLY-$YEAR/KRIG_LONGRANGE

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
#python yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson reanalysis.json
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/WEEKLY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/WEEKLY
python weeklyLowpassSampledError.py --inyear 2018 --inDir $INDIR --outroot $OUTROOT
mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-weekly


# Interpolate a single specific file
export ERRFILE=$OUTROOT/errorfield/stationSummaryAves_03_201801150000_201801210000.csv
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/WEEKLY
python krigListOfErrorSets.py  --cv_kriging  --outroot $OUTROOT --yamlname $YAMLNAME --errorfile $ERRFILE --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-CV-interpolate
