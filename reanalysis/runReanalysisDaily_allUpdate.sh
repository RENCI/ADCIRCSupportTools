#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J DailyFFTReanalysis
#SBATCH --mem-per-cpu 128000

export YEAR=2018
DAILY=DAILY-4MONTH-RANGE2-LP48

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
export OUTROOT=./TESTFREQ
python dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT # --stationarity
mv $RUNTIMEDIR/log $OUTROOT/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=./TESTFREQ
export ERRDIR=$OUTROOT/errorfield
echo "python  iterateKriging.py --daily --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON"
mv $RUNTIMEDIR/log $OUTROOT/log-interpolate
