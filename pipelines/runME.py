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
runProps = os.system('python execute_APSVIZ_pipeline.py --outputDIR /projects/ees/APSViz/stageDIR/insets  --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/nam/2021010500/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc"'/

# EC95d Manually run the new code
#python execute_APSVIZ_pipeline.py --outputDIR /projects/ees/APSViz/stageDIR/insets  --urljson ec95d_data1.json --grid 'ec95d'

python execute_APSVIZ_pipeline.py --urljson ec95d_data1.json --grid 'ec95d'

# REGION3 invocatrion
python execute_APSVIZ_pipeline.py --urljson region3.json --grid 'region3'

# ANother ec956d test 
python execute_APSVIZ_pipeline.py  --urljson datanew.json --grid 'ec95d'
python execute_APSVIZ_pipeline.py  --urljson datanew_hsofs.json --grid 'hsofs'

# Another hsofs terst 
python execute_APSVIZ_pipeline.py  --urljson data2-forecast.json --grid 'hsofs'
# New optional "finalDIR" location storage of PNGs and the CSV
# THis is not the same as using outputDIR and can be used in conjunction with it
python execute_APSVIZ_pipeline.py --urljson ec95d_data1.json --grid 'ec95d' --finalDIR /home/jtilson/ADCIRCSupportTools/FINAL

# Print out dict
print('runProps: {}'.format(runProps))

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/nam/2021050412/ec95d/hatteras.renci.org/ec95d-nam-bob-rptest/namforecast/fort.63.nc" --grid 'ec95d'

# Drop log file into dir instanceId

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/nam/2021050412/ec95d/hatteras.renci.org/ec95d-nam-bob-rptest/namforecast/fort.63.nc" --grid 'ec95d' --instanceId 'CLOUD'

# Example of Florence
python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2018/temp_florence/50/nc_inundation_v9.99_w_rivers/hatteras.renci.org/062018hiresr/nhcConsensus/fort.63.nc" --grid 'nc_inundation_v9.99_w_rivers' --instanceId 'CLOUD'

# Example when not enough nowcasts are avail

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2018/temp_florence/44/nc_inundation_v9.99_w_rivers/hatteras.renci.org/062018hiresr/nhcConsensus/fort.63.nc" --grid 'nc_inundation_v9.99_w_rivers' --instanceId 'CLOUD'



