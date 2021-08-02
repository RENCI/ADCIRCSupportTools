#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000
#SBATCH --exclude=compute-5-17


GRID="region3"
OBSNAME="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/config/obs.region.yml"

export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools
export BASEDIREXTRA=

export KNOCKOUT=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/knockoutStation.json

export YEAR=$1
export RUNTIMEDIR=./REGION3/YEARLY-$YEAR
#export RUNTIMEDIR=./REGION3-DA/YEARLY-$YEAR
export LOG_PATH=$RUNTIMEDIR


URL="/projects/reanalysis/ADCIRC/ERA5/fr3/$YEAR/fort.63.nc"
#URL="/projects/reanalysis/ADCIRC/ERA5/fr3/$YEAR-post/fort.63.nc"

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

export YAMLNAME=$PYTHONPATH/config/int.REANALYSIS.REGION3.yml
export OUTROOT=$RUNTIMEDIR/$DAILY
export ERRDIR=$OUTROOT/errorfield
python  $CODEBASE/runInterpolate_parallel.py  --insill $SILL --inrange $RANGE --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --controlfile $CONTROLFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/AdcircSupportTools.log $OUTROOT/log-interpolate

