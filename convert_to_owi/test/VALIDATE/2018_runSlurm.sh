#!/bin/sh
#SBATCH -t 128:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -J OWI2018
#SBATCH --mem-per-cpu 32000
echo "Begin the OWI processing" 
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH
dir="/home/jtilson/ADCIRCSupportTools/convert_to_owi"
python -u $dir/era5_to_owi.py --year 2018 --convertEastWest True --ddir "/projects/sequence_analysis/vol1/prediction_work/VALIDATE_ERA5/GLOBAL" --odir "/projects/sequence_analysis/vol1/prediction_work/VALIDATE_ERA5/GLOBAL/OWI" --merge False --months "01"
