##
## Simply reformat the slp data provided to Brian
##
import os
import sys
import time
import pandas as pd
from pylab import *
import xarray as xr
import netCDF4
from utilities.utilities import utilities as utilities
import cartopy.crs as ccrs


f='/home/jtilson/ADCIRCSupportTools/compute_error_field/presentation-interpolateScalerField/SINGLE-KRIG-10repetitions/TESTINT0/interpolated/interpolated_wlIOMETA.pkl'

df = pd.read_pickle(f)
df.set_index(['lon','lat'],inplace=True)

##########################################################################

# Convert to Xarray with lon/lat coordinates
dfx = df.to_xarray()

# This is working with the ADRAS virtual machine

p = dfx['value'].T.plot(
    subplot_kws=dict(projection=ccrs.Orthographic(-80, 35), facecolor="gray"),
    transform=ccrs.PlateCarree())
p.axes.set_global()
p.axes.coastlines()
plt.show()

# Grab Brians versiobn which is a better plot

# Works using the ADRAS environment but not the ADCIRCSupport one
# Transpose latitude values
#f='/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/TESTINT-latest/interpolated/interpolated_wl.pkl'
#df = pd.read_pickle(f)
#df.set_index(['lon','lat'],inplace=True)
# Convert to Xarray with lon/lat coordinates
#dfx = df.to_xarray()
#dfx_reindex = dfx.reindex({'lat':dfx['lat'][::-1]})
# This is working with the ADRAS virtual machine
#p = dfx_reindex['value'].T.plot(
#    subplot_kws=dict(projection=ccrs.Orthographic(-100, -50), facecolor="gray"),
#    transform=ccrs.PlateCarree())
#p.axes.set_global()
#p.axes.coastlines()
#plt.show()

