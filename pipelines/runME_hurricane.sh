##
## Example of how APSVIZ might invoke the OBS pipeline
##

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2020/sally/15/LA_v20a-WithUpperAtch_chk/hatteras.renci.org/LAv20a_al192020_jgf/nhcConsensus/fort.63.nc"

# 
python execute_APSVIZ_pipeline.py --grid "ec95d" --inputURL "http://tds.renci.org:8080/thredds/dodsC/2020/cristobal/17/ec95d/supermic.hpc.lsu.edu/ec95d_tropicalcyclone_bde/namforecast/fort.63.nc"

# FLorance is the only one thaty actually has nowcast data to tesat with

python execute_APSVIZ_pipeline.py --grid "ec95d" --urljson data1_hurricaneFlorence_forecast.json

# ANother test 

python execute_APSVIZ_pipeline.py --grid "hsofs" --inputURL "http://tds.renci.org:8080//thredds/dodsC/2021/al08/14/hsofs/hatteras.renci.org/hsofs-al08-bob-2021/nhcOfcl/fort.63.nc"

# IDA test

python execute_APSVIZ_pipeline.py --grid "ec95d" --inputURL "http://tds.renci.org:8080/thredds/dodsC/2021/al09/10/ec95d/hatteras.renci.org/ec95d-al09-bob/veerLeft100/fort.63.nc"

