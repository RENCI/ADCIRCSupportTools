#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J DailyFFTReanalysis
#SBATCH --mem-per-cpu 128000

export YEAR=2018


export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools
#export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools
export PYTHONPATH=$CODEBASE:$PYTHONPATH
export RUNTIMEDIR=.
export BASEDIREXTRA=REANALYSIS-2/YEARLY-$YEAR/KRIG_LONGRANGE

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
#python yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson reanalysis.json
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY
#python dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-daily

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY
export ERRDIR=$OUTROOT/errorfield
python  iterateKriging.py --daily --outroot $OUTROOT --yamlname $YAMLNAME --errordir $ERRDIR --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-interpolate
