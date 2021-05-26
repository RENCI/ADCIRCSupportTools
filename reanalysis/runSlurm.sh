#!/bin/sh
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -J Interpolate
#SBATCH --mem-per-cpu 64000
echo "Begin the Interpolation phase" 
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools
dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis"
python -u $dir/krigListOfErrorSets.py --daily --inrange "8" --insill "0.16" --outroot "./HSOFS/YEARLY-2018/DAILY-hsofs-RANGE8-SILL0.16-NUGGET0.001-LP24" --yamlname "/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/config/int.REANALYSIS.HSOFS.yml" --errorfile "./HSOFS/YEARLY-2018/DAILY-hsofs-RANGE8-SILL0.16-NUGGET0.001-LP24/errorfield/stationSummaryAves_18-365_2018123100.csv" --clampfile "/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/config/clamp_list_hsofs.dat" --gridjsonfile "./HSOFS/YEARLY-2018//adc_coord.json"
