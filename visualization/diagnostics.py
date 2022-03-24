#############################################################
# A collection of methods to facilitate basic checking of ADDA data products.
# Mostly visualization methods that take pre-constructed pkls as input
#
# NOTE: ome of these functions also exist in the individual ADA classes.
# The redundancy is okay for now; it permits testing individual class methods.

# RENCI 2020
#############################################################

# import sys
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import netCDF4

import webbrowser
# import urllib.parse

# from utilities.utilities import utilities

############################################################################
# Some basic functions

print("Matplotlib Version = {}".format(matplotlib.__version__))
print("Numpy Version = {}".format(np.__version__))
print("netCDF4 Version = {}".format(netCDF4.__version__))

###############################################################################
# Station-level diagnostics.


def displayURLsForStations(urlfilename, stationList):
    """
    Read the URL and find the station(s) for loading the proper
    noaa webpages. NOTE: no work done to ensure compatibility with
    multiple browsers.
    df_urls (index +2 column) comes in the form: (station(index), url, value for new)
    """
    if len(stationList) == 0:
        print('No stations provided to url display: quit')
        return
    try:
        df_urls = pd.read_csv(urlfilename, header=0, index_col=0)
    except FileNotFoundError:
        raise IOerror("Failed to read %s" % (config['StationFile']))
    for station in stationList:
        url, new = df_urls.loc[station]
        webbrowser.open(url, new=new)

################################################################################
# Interpolation level diagnostics
# Data processing - graphing

# def addFeatures(ax):
#     ax.add_feature(feature.NaturalEarthFeature(category='cultural',
#                                                name='admin_1_states_provinces_lines',
#                                                scale='50m',
#                                                linewidth=.25,
#                                                facecolor='none',
#                                                edgecolor='black'))
#     ax.add_feature(feature.NaturalEarthFeature(category='physical',
#                                                name='lakes',
#                                                scale='50m',
#                                                linewidth=.25,
#                                                facecolor='none',
#                                                edgecolor='black'))
#     ax.add_feature(feature.NaturalEarthFeature(category='physical',
#                                                name='coastline',
#                                                scale='50m',
#                                                linewidth=.25,
#                                                facecolor='none',
#                                                edgecolor='black'))
#     ax.add_feature(feature.NaturalEarthFeature(category='cultural',
#                                                name='admin_0_boundary_lines_land',
#                                                linewidth=.25,
#                                                scale='50m',
#                                                facecolor='none',
#                                                edgecolor='black'))

# def set_gridLines(ax, xlabels_top, xlabels_bottom, ylabels_left, ylabels_right):
#     gl = ax.gridlines(crs=crs.PlateCarree(), linewidth=1, color='black', alpha=0.5,
#                       linestyle='--')  # , draw_labels=True)
#     gl.xlabels_top = xlabels_top
#     gl.xlabels_bottom = xlabels_bottom
#     gl.ylabels_left = ylabels_left
#     gl.ylabels_right = ylabels_right
#     gl.xlines = True
#     gl.ylines = True
#     gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 15))
#     gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 15))
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER


def plot_interpolation_model(x, y, z, selfX, selfY, selfValues, metadata='Kriging/Python of Matthew Error Vector'):
    """"""
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # , projection=crs.PlateCarree())
    cmap = plt.get_cmap('jet')
    # cmap = cmocean.cm.rain,
    cmap.set_under('gray')
    cax = ax.pcolormesh(x, y, z,
                        cmap=cmap,
                        # transform=crs.PlateCarree(),
                        vmin=-.5, vmax=.5)
    ax.scatter(selfX, selfY, s=100, marker='o',
               c=selfValues, cmap=cmap, edgecolor='k',
               vmin=-.5, vmax=.5)
    fig.colorbar(cax)
    ax.set_xlim([-100, -60])
    ax.set_ylim([5, 50])
    ax.set_title(metadata)
    ax.set_aspect(1.0 / np.cos(np.mean(y) * np.pi / 180.0))
    # ax.set_extent([-91, -69, 19, 46], crs=crs.PlateCarree())
    # addFeatures(ax)
    # set_gridLines(ax, False, True, True, False)
    plt.grid(True)
    # plt.draw()


def plot_scatter_discrete(x, y, values, metadata='Kriging/Python of Matthew Error Vector: Stations',keepfile=False):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # , projection=crs.PlateCarree())
    cmap = plt.get_cmap('jet')
    cmap.set_under('gray')
    cax = ax.scatter(x, y, c=values, s=100, cmap=cmap, vmin=-.5, vmax=.5)
    fig.colorbar(cax)
    ax.set_title(metadata)
    ax.set_aspect(1.0 / np.cos(np.mean(y) * np.pi / 180.0))
    ax.set_xlim([-100, -65])
    ax.set_ylim([20, 50])
    # ax.set_extent([-91, -69, 19, 46], crs=crs.PlateCarree())
    # addFeatures(ax)
    # set_gridLines(ax, False, True, True, False)
    plt.grid(True)
    # plt.draw()

