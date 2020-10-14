#!/bin/sh
#SBATCH -t 128:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -J CDS1994
#SBATCH --mem-per-cpu 32000
echo "Begin the CDS query" 
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH
dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/convert_to_owi"
python -u $dir/cds_request.py --year "1994"
