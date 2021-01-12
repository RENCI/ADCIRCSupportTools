#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000

export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/Reanalysis/ADCIRCSupportTools:$PYTHONPATH
export RUNTIMEDIR=./TEST

python differenceTwoReanalyses.py


