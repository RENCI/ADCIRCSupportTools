##
## Example of how APSVIZ might invoke the OBS pipeline
## The resulting PNGs containg both, the forecast from the last date of the ERR correction AND
## A prepend of the nowcast ENDING at the timein of the error correction
##

## Run the compError starting at 
## FOR SIMPLICITY SIMPLY construnt the correct URL and execute_APSVIZ_pipeline.py will do the rest up 'till 
## the prepend


#1 Build a URL fort passing 

python execute_APSVIZ_pipeline.py --url 'http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020110100/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'


# Now build a second json that will be used for prepending
# timein='2020-10-26'
# For this invocation must use the forecast yaml config
# The variableName is help to allow the subsequence PNG assembler to work right

export RUNTIMEDIR='./NEWTEST2'
python ../get_adcirc/GetADCIRC.py --timeout '2020-10-26 00:00' --writeJson --variableName ADCPrepend --adcYamlname /home/jtilson/ADCIRCSupportTools/config/adc_forecast.yml

#
# Now build the new PNGs
#
/home/jtilson/ADCIRCSupportTools/pipelines/NEWTEST2/ADCIRC
