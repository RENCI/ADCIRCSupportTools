#!/bin/sh
#SBATCH -t 128:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -J OWI2018
#SBATCH --mem-per-cpu 64000
echo "Begin the OWI processing" 
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH
dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/convert_to_owi"
python -u $dir/era5_to_owi.py --metafilename "01" --metadirname "2018" --convertEastWest --ddir "/projects/ees/TDS/ERA5/global" --date_start "201801" --date_end "201801"
