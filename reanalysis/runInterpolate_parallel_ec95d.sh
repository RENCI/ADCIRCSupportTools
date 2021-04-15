#!/bin/sh
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH -J EC95DReanalysis
#SBATCH --mem-per-cpu 64000

export YEAR=$1
export SILL=0.015
export RANGE=8
export NUGGET=0.005
export NLAGS=6

GRID="ec95d"
URL="/projects/ees/TDS/Reanalysis/ADCIRC/ERA5/ec95d/$YEAR/fort.63.nc"


#DAILY=DAILY-2018YEAR-12MONTH-REGION3-RANGE$RANGE-SILL$SILL-NUGGET$NUGGET-LP48
DAILY=DAILY-$YEAR-YEAR-12MONTH-REGION3-RANGE$RANGE-SILL$SILL-NUGGET$NUGGET-LP48

echo $YEAR
echo $DAILY

export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools
export RUNTIMEDIR=./EC95D/YEARLY-$YEAR
export BASEDIREXTRA=


# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
python $CODEBASE/yearlyReanalysis.py --grid $GRID --url $URL
mv $RUNTIMEDIR/AdcircSupportTools.log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/
export OUTROOT=$RUNTIMEDIR/$DAILY
#python $CODEBASE/dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT # --stationarity
#mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$DAILY
export ERRDIR=$OUTROOT/errorfield
#python  $CODEBASE/runInterpolate_parallel.py  --insill $SILL --inrange $RANGE --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
#mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-interpolate
