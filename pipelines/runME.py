##
## Example of how APSVIZ might invoke the OBS pipeline
##

import os


#1 Build a URL fort passing 
#url='http://tds.renci.org/thredds//dodsC/2020/nam/2020092118/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'
url='http://tds.renci.org/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'

# URL test Launch the job using default directory metadata
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc"')

# URL test Launch the job using default directory metadata
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "~/vol1/prediction_work/Reanalysis/LOCAL_NC_DATA/fort.63.nc"')

#APSVIZ test example
runProps = os.system('python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021010500/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc"')

#APSVIZ test example save to alternative location
runProps = os.system('python execute_APSVIZ_pipeline.py --outputDIR /projects/ees/APSViz/stageDIR/insets  --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021010500/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc"'/

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

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021050412/ec95d/hatteras.renci.org/ec95d-nam-bob-rptest/namforecast/fort.63.nc" --grid 'ec95d'

# Drop log file into dir instanceId

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021050412/ec95d/hatteras.renci.org/ec95d-nam-bob-rptest/namforecast/fort.63.nc" --grid 'ec95d' --instanceId 'CLOUD'

# Example of Florence
python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2018/temp_florence/58/nc_inundation_v9.99_w_rivers/hatteras.renci.org/062018hiresr/nhcConsensus/fort.63.nc" --grid 'nc_inundation_v9.99_w_rivers' --instanceId 'CLOUD'

# Example when not enough nowcasts are avail

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2018/temp_florence/44/nc_inundation_v9.99_w_rivers/hatteras.renci.org/062018hiresr/nhcConsensus/fort.63.nc" --grid 'nc_inundation_v9.99_w_rivers' --instanceId 'CLOUD'


# Example from apsviz2:

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021052106/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc" --grid 'hsofs'


# A previously failed example

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021052318/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc" --grid 'hsofs'

# A recent APSVIZ run where the number of stations seems to have decreased.

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021060912/ec95d/hatteras.renci.org/ec95d-nam-bob-postNowcast/namforecast/fort.63.nc" --grid "ec95d"

# Try the new grid.

python execute_APSVIZ_pipeline.py --inputURL "https://fortytwo.cct.lsu.edu/thredds/dodsC/2021/al03/06/LA_v20a-WithUpperAtch_chk/supermic.hpc.lsu.edu/LAv20a_al032021_jgf_23kcms/nhcConsensus/fort.63.nc" --grid "LA_v20a-WithUpperAtch_chk"

python execute_APSVIZ_pipeline.py --inputURL "https://fortytwo.cct.lsu.edu/thredds/dodsC/2021/al03/10/LA_v20a-WithUpperAtch_chk/supermic.hpc.lsu.edu/LAv20a_al032021_jgf_23kcms/nhcConsensus/fort.63.nc" --grid "LA_v20a-WithUpperAtch_chk"


python execute_APSVIZ_pipeline.py --inputURL 'http://tds.renci.org/thredds/dodsC/2021/nam/2021072606/nc_inundation_v9.99_w_rivers/hatteras.renci.org/ncv99-nam-bob-2021/namforecast/fort.63.nc'

# Test the uriv18 grid

# A new grid test

python execute_APSVIZ_pipeline.py --grid "uriv18" --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021082000/uriv18/hatteras.renci.org/uriv18-nam-bob-2021/namforecast/fort.63.nc"

# A data set rerun

python execute_APSVIZ_pipeline.py --grid "hsofs" --inputURL "http://tds.renci.org/thredds/dodsC/2021/al08/22/hsofs/hatteras.renci.org/hsofs-al08-bob-2021/nhcOfcl/fort.63.nc"

# 

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021060912/ec95d/hatteras.renci.org/ec95d-nam-bob-postNowcast/namforecast/fort.63.nc" --grid "ec95d"

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021091712/NCSC_SAB_v1.15/hatteras.renci.org/ncsc115-nam-2021/namforecast/fort.63.nc" --grid "NCSC_SAB_v1.15"


python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2021/nam/2021092018/NCSC_SAB_v1.15/hatteras.renci.org/ncsc115-nam-2021/namforecast/maxele.63.nc" --grid "NCSC_SAB_v1.15"

python execute_APSVIZ_pipeline.py --inputURL "http://adcircvis.tacc.utexas.edu/thredds/dodsC/asgs/2021/nam/2021092306/SABv20a/frontera.tacc.utexas.edu/SABv20a_nam_jgf_status/namforecast/fort.63.nc" --grid "SABv20a"

# New test: comparer passing a namforecast vs a nowcast

# Forecast
python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2022/nam/2022012618/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/namforecast/fort.63.nc" --grid 'hsofs'

# Nowcast 
python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2022/nam/2022012618/hsofs/hatteras.renci.org/hsofs-nam-bob-2021/nowcast/fort.63.nc" --grid 'hsofs'

# Grid test: NCSC_SAB_V1.20
python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org/thredds/dodsC/2022/nam/2022021518/NCSC_SAB_v1.20/hatteras.renci.org/ncsc120-nam-2022/namforecast/fort.63.nc --grid 'NCSC_SAB_v1.20'


