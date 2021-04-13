##
## Example of how APSVIZ might invoke the OBS pipeline
##

import os


#1 Build a URL fort passing 
#url='http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020092118/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'
url='http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'

# URL test Launch the job using default directory metadata
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc"')

# URL test Launch the job using default directory metadata
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "~/vol1/prediction_work/Reanalysis/LOCAL_NC_DATA/fort.63.nc"')

#APSVIZ test example
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/nam/2021010500/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc"')

#APSVIZ test example save to alternative location
runProps = os.system('python execute_APSVIZ_pipeline.py --outputDir /projects/ees/APSViz/stageDIR/insets  --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/nam/2021010500/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc"')

# EC95d Manually run the new code
#python execute_APSVIZ_pipeline.py --outputDir /projects/ees/APSViz/stageDIR/insets  --urljson ec95d_data1.json --grid 'ec95d'

python execute_APSVIZ_pipeline.py --urljson ec95d_data1.json --grid 'ec95d'

# REGION3 invocatrion
python execute_APSVIZ_pipeline.py --urljson region3.json --grid 'region3'

# New optional "final" location storage of PNGs and the CSV
# THis is not the same as using outputDir and can be used in conjunction with it
python execute_APSVIZ_pipeline.py --urljson ec95d_data1.json --grid 'ec95d' --final /home/jtilson/ADCIRCSupportTools/FINAL

# Print out dict
print('runProps: {}'.format(runProps))
