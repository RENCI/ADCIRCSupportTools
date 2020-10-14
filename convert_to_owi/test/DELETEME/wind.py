##
## Cannot call for a years worth at one time.
## So here we simply chop it up into single month retrieves.
## Based on the API call constructed by copernicus when requesting a year of data
## They specify the months but then only specify 31 days. So presumably, this does not
## cause a failure for months with < 31 days

## Also assume we need to reconnect each month

import os,sys
import pandas as pd
import cdsapi

###############################################################################
def wrap_retrieve(month, filename):
    print('Start month: {} outfilename is {}'.format(month, filename))
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
            ],
            'month': [ '01' ],
            'year': '2018',
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        filename)
## end defs
###############################################################################

monthList = ['01','02','03','04','05','06','07','08','09','10','11','12']
monthList = ['01']

for month in monthList:
    outfilename = '_'.join([month,'2018','download_wind.nc'])
    wrap_retrieve(month, outfilename)

print('Finished')

