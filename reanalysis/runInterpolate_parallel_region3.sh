#!/bin/sh

#export YEAR=2018
export YEAR=2018DA

export SILL=0.015
export RANGE=8
export NUGGET=0.005

# /projects/ees/TDS/Reanalysis/ADCIRC/ERA5/fr3/2018/fort.63.nc

#DAILY=DAILY-4MONTH-RANGE$RANGE-SILL$SILL-LP48
DAILY=DAILY-2018YEAR-12MONTH-REGION3-RANGE$RANGE-SILL$SILL-NUGGET$NUGGET-LP48

export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis
export PYTHONPATH=$CODEBASE
export RUNTIMEDIR=.
export BASEDIREXTRA=REANALYSIS_COMPREHENSIVE_REGION3/YEARLY-$YEAR

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
#python $CODEBASE/yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson region3.json --grid region3
python $CODEBASE/yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson region3_da.json --grid region3
mv $RUNTIMEDIR/AdcircSupportTools.log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/$DAILY
python $CODEBASE/dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT # --stationarity
mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/$DAILY
export ERRDIR=$OUTROOT/errorfield
python  $CODEBASE/runInterpolate_parallel.py  --insill $SILL --inrange $RANGE --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-interpolate
