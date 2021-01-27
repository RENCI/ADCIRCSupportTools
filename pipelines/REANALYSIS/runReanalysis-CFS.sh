#!/bin/sh
#SBATCH -t 512:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH -J Reanalysis
#SBATCH --mem-per-cpu 512000

export PYTHONPATH=~/ADCIRCSupportTools:$PYTHONPATH
export RUNTIMEDIR=.

python yearlyReanalysis.py --urljson reanalysis-cfsv2.json


