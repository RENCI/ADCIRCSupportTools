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

infd = '/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/slp2050_reformat.csv'
dfin = pd.read_csv(infd)
dfin['mean'].hist()

#f = '/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/TESTINT/interpolated/interpolated_wl.pkl'
f = '/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/CV_OPTIMIZE/interpolated/interpolated_wl.pkl'
#f = '/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/CV_OPTIMIZE_FIRST/interpolated/interpolated_wl.pkl'
df = pd.read_pickle(f)
df.set_index(['lon','lat'],inplace=True)

##########################################################################
# Are the INPUT values normally distributed ?
#
# THe data are a little left skewed and perhaps need to be noamalized


# Convert to Xarray with lon/lat coordinates
dfx = df.to_xarray()

# This is working with the ADRAS vertualk machine
# The TRANSPOSE was important :(

p = dfx['value'].T.plot(
    subplot_kws=dict(projection=ccrs.Orthographic(-80, 35), facecolor="gray"),
    transform=ccrs.PlateCarree())
p.axes.set_global()
p.axes.coastlines()
plt.show()

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


#####################################################################
## Plot input data with no interpolation
slpf = '/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/slp2050_reformat.csv'
dfin = pd.read_csv(slpf, header=0, index_col=0)
dfin.set_index(['lon','lat'],inplace=True)
dfin.columns=['value']
dfin['value'].hist()


dfxin = dfin['mean'].to_xarray() # sets all means to nan wtf!
# values get reset to nans. wtf !

p = dfxin.T.plot(
    subplot_kws=dict(projection=ccrs.Orthographic(-80, 35), facecolor="gray"),
    transform=ccrs.PlateCarree())
p.axes.set_global()
p.axes.coastlines()
plt.show()

###########################################################################

# Hist() of the dataframe of the data
# CODE below here only draws me blank images

###########################################################################
##
## COde from Brians notebook
##

import pandas as pd
import numpy as np
import xarray as xr
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

f='/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/TESTINT/interpolated/interpolated_wl.pkl'
d = pd.read_pickle(f)
d.keys()

d=pd.read_pickle('/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/TESTINT-smoothed/interpolated/interpolated_wl.pkl')
d.keys()

lon = np.unique(d.lon)
lat = np.unique(d.lat)
z = np.reshape(d.values[:,2],(lat.shape[0],lon.shape[0])).T
print(lon.shape, lat.shape, z.shape)

ds = xr.Dataset()
ds["z"] = (("lon", "lat"), z)
ds.coords["lat"] = (("lat"), lat)
ds.coords["lon"] = (("lon"), lon)

crs = ccrs.LambertConformal(central_longitude=-80, central_latitude=36)

# Function used to create the map subplots
def plot_background(ax):
    #ax.set_extent([235., 290., 20., 55.])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), constrained_layout=True,
                       subplot_kw={'projection': crs})
plot_background(ax)


cf=ds['z'].T.plot(transform=ccrs.PlateCarree(),add_colorbar=False)
cb1 = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.65, pad=0)
ax.coastlines()
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')







