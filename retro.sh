#!/bin/bash

export ADDAHOME=/Users/bblanton/ADCIRCDataAssimilation
export RUNTIMEDIR="./work"
export PYTHONPATH=/Users/bblanton/.venvburrito/lib/python2.7/site-packages::/Users/bblanton/GitHub/RENCI/ADCIRCDataAssimilation/
PYTHON=/Users/bblanton/.virtualenvs/adcirc_DA/bin/python

Ystart=2020
Mstart=03
Dstart=17
CYstart=00
CYstart=`echo "6*($CYstart/6)" | bc`

Yend=2020
Mend=03
Dend=20
CYend=06
CYend=`echo "6*($CYend/6)" | bc`

Tstart=$(gdate -u +"%s" --date="$Ystart/$Mstart/$Dstart")
Tend=$(gdate -u +"%s" --date="$Yend/$Mend/$Dend")
Tnow=$Tstart
Tinc=21600

echo $Tstart
echo $Tend

#YYYYmmdd-hh:mm:ss

while [ $Tnow -le $Tend ] ; do 
	temp=$(gdate -u -d @$Tnow +"%Y%m%d-%H:%M:%S")
	echo Running ADDA for $temp, $Tnow
	$PYTHON -u ADDA/ADDA_withCLI.py --error_histograms False --vis_error False --time2 $temp 2>&1 | tee  $RUNTIMEDIR/log
	# add cycle increment
	Tnow=$((Tnow+Tinc))
done
