##
## Example of how APSVIZ might invoke the OBS pipeline
##

import os


#1 Build a URL fort passing 
#url='http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020092118/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'
url='http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc'

# Launch the job using default directory metadata
runProps = os.system('python execute_APSVIZ_pipeline.py --url "http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020110618/hsofs/hatteras.renci.org/hsofs-nam-bob/namforecast/fort.63.nc"')

# Print out dict
print('runProps: {}'.format(runProps))
