#!/bin/sh

export YEAR=2018
export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools
export PYTHONPATH=$CODEBASE:$PYTHONPATH
export RUNTIMEDIR=.
export BASEDIREXTRA=REANALYSIS-2/YEARLY-$YEAR/KRIG_LONGRANGE


# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY
export ERRDIR=$OUTROOT/errorfield

python  runInterpolate_parallel.py --daily --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-interpolate
