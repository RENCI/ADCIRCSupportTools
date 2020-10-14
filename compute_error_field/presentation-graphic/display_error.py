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
import cartopy.feature as cfeature


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

#'/projects/sequence_analysis/vol1/prediction_work/EDS_OceanRise/BOX_CLAMPS/TESTINT-smoothed/interpolated/interpolated_wl.pkl'e## 
# Brians approach

d=pd.read_pickle(f)
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

# GIves a blank image

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([5, 5, 10, 10])

p = ds['z'].T.plot(
    subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor="gray"),
    transform=ccrs.PlateCarree())
p.axes.set_global()
p.axes.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(states_provinces, edgecolor='gray')
plt.show()




