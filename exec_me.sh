#!/bin/bash

export ADDAHOME=/Users/bblanton/ADCIRCDataAssimilation
export RUNTIMEDIR="./work"
export PYTHONPATH=/Users/bblanton/.venvburrito/lib/python2.7/site-packages::/Users/bblanton/GitHub/RENCI/ADCIRCDataAssimilation/
PYTHON=/Users/bblanton/.virtualenvs/adcirc_DA/bin/python
SLEEP=3600

while [ 1 ] ; do 
	NOW=$(date -u  +"%Y-%m-%d_Z_%T")
	$PYTHON -u ADDA/ADDA_withCLI.py --error_histograms False  --vis_error True  2>&1 | tee  $RUNTIMEDIR/log.$NOW 
        sleep $SLEEP
done
