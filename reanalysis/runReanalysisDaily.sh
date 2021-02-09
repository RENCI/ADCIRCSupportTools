#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J DailyFFTReanalysis
#SBATCH --mem-per-cpu 512000

export YEAR=2018


export CODEBASE=/projects/sequence_analysis/vol1/prediction_work/CausalInference/CausalNetworking_forKirk/TEST/ADCIRCSupportTools
export PYTHONPATH=$CODEBASE:$PYTHONPATH
export RUNTIMEDIR=.
export BASEDIREXTRA=TESTFULL/STATE/YEARLY-2018/KRIG_LONGRANGE

# Build the yearly error file store in $RUNTIMEDIR/BASEDIREXTRA
#python yearlyReanalysis.py --iosubdir $BASEDIREXTRA --urljson reanalysis.json
#mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-yearly

# Store files in $RUNTIMEDIR/DAILY/errorfield
export INDIR=$RUNTIMEDIR/$BASEDIREXTRA
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY
python dailyLowpassSampledError.py --inDir $INDIR --outroot $OUTROOT
mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-weekly


# Interpolate a single specific file
#export ERRFILE=$OUTROOT/errorfield/stationSummaryAves_18-332_2018112800.csv
#export ADCJSON=$INDIR/adc_coord.json
#export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
#export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
#export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY

# Interpolate a single specific file
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY
export ERRDIR=$OUTROOT/errorfield

export ERRFILE=$OUTROOT/errorfield/stationSummaryAves_18-332_2018112800.csv
export ADCJSON=$INDIR/adc_coord.json
export CLAMPFILE=$CODEBASE/config/clamp_list_hsofs.dat
export YAMLNAME=$CODEBASE/config/int.REANALYSIS.yml
export OUTROOT=$RUNTIMEDIR/$BASEDIREXTRA/DAILY

#python krigListOfErrorSets.py  --outroot $OUTROOT --yamlname $YAMLNAME --errorfile $ERRFILE --clampfile $CLAMPFILE --gridjsonfile $ADCJSON
python krigListOfErrorSets.py  --daily --outroot $OUTROOT --yamlname $YAMLNAME --errorfile $ERRFILE --clampfile $CLAMPFILE --gridjsonfile $ADCJSON

mv $RUNTIMEDIR/log $RUNTIMEDIR/$BASEDIREXTRA/log-interpolate
