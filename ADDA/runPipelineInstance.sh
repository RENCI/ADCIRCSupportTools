#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=64000

module load python/3.7.0
echo $HOSTNAME

# assumes ADDAHOME is set 
# Set RUNTIMEDIR to whatever you like
ADDAHOME="/home/jtilson/ADCIRCDataAssimilation"
PYTHONPATH=$ADDAHOME:$PYTHONPATH
RUNTIMEDIR="/projects/sequence_analysis/vol1/prediction_work/TESTADDA"
export PYTHONPATH
export ADDAHOME
export RUNTIMEDIR

source $ADDAHOME/venv/bin/activate
python -u $ADDAHOME/ADDA/ADDA_withCLI.py --vis_scatter False  --error_histograms False --override_repeats True
source $ADDAHOME/venv/bin/deactivate
