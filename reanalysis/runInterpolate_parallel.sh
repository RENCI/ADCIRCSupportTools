#!/bin/sh

export RANGE=2
export YEAR=2018
DAILY=DAILY-4MONTH-RANGE$RANGE-LP48

export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools
#export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools
export PYTHONPATH=$CODEBASE:$PYTHONPATH
export RUNTIMEDIR=.
export BASEDIREXTRA=REANALYSIS_COMPREHENSIVE/YEARLY-$YEAR

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
#python yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson reanalysis.json
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/$DAILY
python dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT # --stationarity
mv $RUNTIMEDIR/log $OUTROOT/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/$DAILY
export ERRDIR=$OUTROOT/errorfield
python  runInterpolate_parallel.py  --daily --inrange $RANGE --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/log $OUTROOT/log-interpolate

