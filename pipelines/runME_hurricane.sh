##
## Example of how APSVIZ might invoke the OBS pipeline
##

python execute_APSVIZ_pipeline.py --inputURL "http://tds.renci.org:8080/thredds/dodsC/2020/sally/15/LA_v20a-WithUpperAtch_chk/hatteras.renci.org/LAv20a_al192020_jgf/nhcConsensus/fort.63.nc"

# 
python execute_APSVIZ_pipeline.py --grid "ec95d" --inputURL "http://tds.renci.org:8080/thredds/dodsC/2020/cristobal/17/ec95d/supermic.hpc.lsu.edu/ec95d_tropicalcyclone_bde/namforecast/fort.63.nc"

# FLorance is the only one thaty actually has nowcast data to tesat with

python execute_APSVIZ_pipeline.py --grid "ec95d" --urljson data1_hurricaneFlorence_forecast.json

