#!/bin/sh
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH -J EC95DReanalysis
#SBATCH --mem-per-cpu 64000
#SBATCH --exclude=compute-7-30

export SILL=0.16
export RANGE=8
export NUGGET=0.001
export NLAGS=6

GRID="ec95d"
OBSNAME="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/config/obs.ec95d.yml"

export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools
export BASEDIREXTRA=
export KNOCKOUT=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/knockoutStation.json

export YEAR=$1
export RUNTIMEDIR=./EC95D/YEARLY-$YEAR
#export RUNTIMEDIR=./EC95D-DA/YEARLY-$YEAR
export LOG_PATH=$RUNTIMEDIR

URL="/projects/reanalysis/ADCIRC/ERA5/ec95d/$YEAR/fort.63.nc"
#URL="/projects/reanalysis/ADCIRC/ERA5/ec95d/$YEAR-post/fort.63.nc"

echo $URL

DAILY=DAILY-$GRID-LP24

echo "xxxxxx"
echo $YEAR
echo $DAILY
echo $RUNTIMEDIR
echo "xxxxxx"

####

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
python $CODEBASE/yearlyReanalysisRoundHourly.py --obsfile $OBSNAME --grid $GRID --url $URL --knockout $KNOCKOUT
mv $RUNTIMEDIR/AdcircSupportTools.log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/
export OUTROOT=$RUNTIMEDIR/$DAILY
python $CODEBASE/dailyLowpassSampledError.py --inyear $YEAR  --inDir $INDIR --outroot $OUTROOT # --stationarity
mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$PYTHONPATH/config/water_control_list.dat
export CONTROLFILE=$PYTHONPATH/config/land_control_list.dat
# clamps are disabled in the code itself
export YAMLNAME=$PYTHONPATH/config/int.REANALYSIS.EC95D.yml
export OUTROOT=$RUNTIMEDIR/$DAILY
export ERRDIR=$OUTROOT/errorfield
#python  $CODEBASE/runInterpolate_parallel.py  --insill $SILL --inrange $RANGE --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --controlfile $CONTROLFILE --gridjsonfile $ADCJSON
python  $CODEBASE/runInterpolate_parallel.py  --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --controlfile $CONTROLFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-interpolate

