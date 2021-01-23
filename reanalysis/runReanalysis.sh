#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000

export PYTHONPATH=export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools:$PYTHONPATH
export RUNTIMEDIR=./YEARLY

# Compute the error file
python yearlyReanalysis.py --urljson reanalysis.json
w
# Store files in $RUNTIMEDIR/WEEKLY/errorfield
python weeklyLowpassSampledError.py --inyear 2018 --yearlyDir $RUNTIMEDIR




