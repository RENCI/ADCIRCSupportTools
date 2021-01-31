#!/usr/bin/env python
# coding: utf-8

# This code originally derived from a jhub notebook 
# implements an approach for computing geotiffs from ADCIRC FEM output in netCDF.

# If inputing url as a json then we must have filename metadata included as:
#{"1588269600000": "http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020043018/hsofs/hatteras.renci.org/ncfs-dev-hsofs-nam-master/nowcast/maxele.63.nc"}
# If simply imputting a raw url (not expected to be a common approach) then the utime value will be set to 0000

import os
import sys
import time
import datetime as dt
import numpy.ma as ma
import pandas as pd

from pylab import *
import matplotlib.tri as Tri
from matplotlib import pyplot
import netCDF4

from utilities.utilities import utilities as utilities

import rasterio as rio
from rasterio.transform import from_origin
from rasterio.plot import show
import geopandas as gpd

utilities.log.info('Begin TIF generation')
utilities.log.info("netCDF4 Version = {}".format(netCDF4.__version__))
utilities.log.info("Pandas Version = {}".format(pd.__version__))
utilities.log.info("Matplotlib Version = {}".format(matplotlib.__version__))
utilities.log.info("rasterio Version = {}".format(rio.__version__))
utilities.log.info("Geopandas Version = {}".format(gpd.__version__))

# define url functionality 
# http://tds.renci.org:8080/thredds/dodsC/2020/nam/2020012706/hsofs/hatteras.renci.org/ncfs-dev-hsofs-nam-master/namforecast/maxele.63.nc

ensname = 'namforecast'

def checkEnumuation(v):
    if v.lower() in ('zeta_max'):
        return True
    if v.lower() in ('vel_max'):
        return True
    if v.lower() in ('inun_max'):
        return True
    return False

def get_url(dict_cfg, varname, datestr, hourstr, yearstr, enstag):
    """
    Build a URL for finding ADCIRC information.
    Parameters:
        datastr: "YYYYmmdd"
        hourstr: "hh"
        yearstr: "YYYY"
        input dict.
    Returns:
        url as a full path string
    """
    # dstr = dt.datetime.strftime(d, "%Y%m%d%H")
    url = dict_cfg['baseurl']+dict_cfg['dodsCpart'] % \
        (yearstr,
         datestr+hourstr,
         dict_cfg['AdcircGrid'],
         dict_cfg['Machine'],
         dict_cfg['Instance'],
         enstag,
         varname)
    return url


def validate_url(url):  # TODO
    status = True  # Innocent until proven guilty
    # check for validity
    if not status:
        utilities.log.error("Invalid URL:".format(url))
    return status 

# Define some basic grid functionality and defaults

# If the ASGS serbver is buggered but thredds is okay, this can results
# in a fault thaty cannot easily be trapped:
# Example: 
# <class 'netCDF4._netCDF4.Variable'>
# dim->dim.declsize0 > 0

def get_adcirc_grid(nc):
    agdict = {}
    agdict['lon'] = nc.variables['x'][:]
    agdict['lat'] = nc.variables['y'][:]
    # nv is the triangle list
    agdict['ele'] = nc.variables['element'][:,:] - 1
    agdict['latmin'] = np.mean(nc.variables['y'][:])  # needed for scaling lon/lat plots
    return agdict


def get_adcirc_slice(nc, v, it=None):
    advardict = {}
    var = nc.variables[v]
    if re.search('max', v):
        var_d = var[:]  # the actual data
    else:
        var_d = var[it, :]  # the actual data
    var_d[var_d.mask] = np.nan
    advardict['data'] = var_d
    return advardict


def default_inter_grid():
    """
    A simple grid for testing. See also read_inter_grid
    to fetch grid data from the yaml
    Returns:
        targetgrid
        target crs 
    """
    upperleft_lo = -77.
    upperleft_la = 35.7
    res = 100  # resolution in meters
    nx = 1000
    ny = 1000
    crs = 'epsg:6346'  
    targetgrid = {'Latitude': [upperleft_la],
                  'Longitude': [upperleft_lo],
                  'res': res,
                  'nx': nx,
                  'ny': ny}
    return targetgrid, crs


def read_inter_grid_yaml(geo_yamlfile=os.path.join( os.path.dirname(__file__), '../config', 'geotiff.yml')):
    """
    fetch geotiff grid parameters from the yaml
    Returns:
        targetgrid dict
        target crs 
    """
    try:
        config = utilities.load_config(geo_yamlfile)['REGRID']
    except:
        utilities.log.error("REGRID: load_config yaml failed")
    upperleft_lo = config['upperleft_lo']
    upperleft_la = config['upperleft_la']
    res = config['res']
    nx = config['nx']
    ny = config['ny']
    #crs = config['coordrefsys']
    crs = config['reference']
    targetgrid = {'Latitude': [upperleft_la],
                  'Longitude': [upperleft_lo],
                  'res': res,
                  'nx': nx,
                  'ny': ny}
    return targetgrid, crs

# Define geopandas processors
# project grid coords, before making Triangulation object
def construct_geopandas(agdict, targetepsg):
    # print('Making DataFrame of ADCIRC Grid Coords')
    df_Adcirc = pd.DataFrame(
        {'Latitude': agdict['lat'],
        'Longitude': agdict['lon']})
    utilities.log.info('Converting to Geopandas DF')
    t0 = time.time()
    gdf = gpd.GeoDataFrame(
        df_Adcirc, geometry=gpd.points_from_xy(agdict['lon'], agdict['lat']))
    # print(gdf.crs)
    # init crs is LonLat, WGS84
    utilities.log.info('Adding WGS84 crs')
    gdf.crs = {'init':'epsg:4326'}
    utilities.log.info('Converting to {}'.format(targetepsg))
    gdf = gdf.to_crs({'init': targetepsg})
    # get converted ADCIRC node coordinates
    xtemp = gdf['geometry'].x
    ytemp = gdf['geometry'].y
    utilities.log.info('Time to create geopandas was {}'.format(time.time()-t0))
    utilities.log.debug('GDF data set {}'.format(gdf))
    return xtemp, ytemp, gdf

# project interpolation grid to target crs
def compute_geotiff_grid(targetgrid, targetepsg):
    """
    Results:
        meshdict. Values for upperleft_x, upperleft_y, x,y,xx,yy,xxm,yym
    """
    df_target = pd.DataFrame(data=targetgrid)
    gdf_target = gpd.GeoDataFrame(
        df_target, geometry=gpd.points_from_xy(df_target.Longitude, df_target.Latitude))
    # print(gdf_target.crs)

    # init projection is LonLat, WGS84
    gdf_target.crs = {'init': 'epsg:4326'}

    # convert to "targetepsg"
    gdf_target = gdf_target.to_crs({'init': targetepsg})
    # print(gdf_target.crs)

    # compute spatial grid for raster (for viz purposes below)
    upperleft_x = gdf_target['geometry'][0].x
    upperleft_y = gdf_target['geometry'][0].y
    x = np.arange(upperleft_x, upperleft_x+targetgrid['nx']*targetgrid['res'], targetgrid['res'])
    y = np.arange(upperleft_y, upperleft_y-targetgrid['ny']*targetgrid['res'], -targetgrid['res'])
    xx, yy = np.meshgrid(x, y)

    # get centroid coords
    xm = (x[1:] + x[:-1]) / 2
    ym = (y[1:] + y[:-1]) / 2
    xxm, yym = np.meshgrid(xm, ym)
    utilities.log.debug('compute_mesh: lon {}. lat {}'.format(upperleft_x, upperleft_y))
    meshdict = {'uplx': upperleft_x,
                'uply': upperleft_y,
                'x': x,
                'y': y,
                'xx': xx,
                'yy': yy,
                'xxm': xxm,
                'yym': yym}

    return meshdict

# Aggregation of individual methods

def construct_url(varname, geo_yamlfile=os.path.join(os.path.dirname(__file__), '../config', 'geotiff.yml')):
    """
    Assembles several method into an aggregate method to grab parameters
    from the yaml and construct a url. This is skipped if the user
    specified a URL on input.
    """
    geo_yml= utilities.load_config(geo_yamlfile)
    varnamedict = geo_yml['VARFILEMAP'] 
    varfile = varnamedict[varname]
    utilities.log.info('map dict {}'.format(varnamedict))
    # Specify time parameters of interest
    timedict = geo_yml['TIME'] 
    doffset = timedict['doffset']
    hoffset = timedict['hoffset']

    # Set time for url
    thisdate = dt.datetime.utcnow() + dt.timedelta(days=doffset) + dt.timedelta(hours=hoffset)
    cyc = "%02d" % (6 * int(thisdate.hour / 6))     # Hour/cycle specification
    dstr = dt.datetime.strftime(thisdate, "%Y%m%d") 
    ystr = dt.datetime.strftime(thisdate, "%Y")  # NOTE slight API change to get_url

    # Fetch url
    urldict = geo_yml['ADCIRC']
    utilities.log.info('url dict {}'.format(urldict))

    # TODO: we will also need to elevate "namforecast" to be a default/input parameter, since this
    # can take several different values that depend in the ASGS configuration.
    # However, namforecast is a reasonable default
    url = get_url(urldict, varfile, dstr, str(cyc), ystr, ensname)
    utilities.log.info('Validated url {}'.format(url))
    utilities.log.info('Datetime {}'.format(dstr))
    utilities.log.debug('Constructed URL {}'.format(url))
    return url, dstr, cyc


# get ADCIRC grid parts;  this need only be done once, as it can be time-consuming over the network
def extract_grid(url, targetepsg):
    """
    Extract ADCIRC grid parts from THREDDS dataset
    Results:
        tri:
    """
    nc = netCDF4.Dataset(url)
    # print(nc.variables.keys())
    agdict = get_adcirc_grid(nc)

    xtemp, ytemp, gdf = construct_geopandas(agdict, targetepsg)
    tri = Tri.Triangulation(xtemp, ytemp, triangles=agdict['ele'])
    return tri


# Assemble some optional plot methods
def plot_triangular_vs_interpolated(meshdict, varname, tri, zi_lin, advardict):
    """
    A rough plotting routine generally used for validation studies.
    The real data results are the tif fle that gets generated later
    """
    xm0, ym0 = meshdict['uplx'], meshdict['uply']
    x, y = meshdict['x'], meshdict['y']
    xx, yy = meshdict['xx'], meshdict['yy']
    xxm, yym = meshdict['xxm'], meshdict['yym']
    nlev = 11
    vmin = np.floor(np.nanmin(zi_lin))
    vmax = np.ceil(np.nanmax(zi_lin))
    levels = linspace(0., vmax, nlev+1)
    utilities.log.info('Levels are {}, vmin {}, vmax {}'.format(levels, vmin, vmax))
    #
    v = advardict['data']
    utilities.log.debug('nanmin {}, nammix {}'.format(np.nanmin(v),np.nanmax(v))) 
    #
    cmap = plt.cm.get_cmap('jet', 8)
    # Start the plots
    fig, ax = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)
    # tcf = ax[0].tricontourf(tri, v,cmap=plt.cm.jet,levels=levels)
    if True:
        # fig, ax = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)
        tcf = ax[0].tripcolor(tri, v, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        ax[0].set_aspect('equal')
        ax[0].plot(xx[0, ], yy[0, ], color='k', linewidth=.25)
        ax[0].plot(xx[-1, ], yy[-1, ], color='k', linewidth=.25)
        ax[0].plot(xx[:, 0], yy[:, 0], color='k', linewidth=.25)
        ax[0].plot(xx[:, -1], yy[:, -1], color='k', linewidth=.25)
        fig.colorbar(tcf, ax=ax[0], orientation='horizontal')
        ax[0].set_title('ADCIRC {}'.format(varname), fontsize=14)
    pcm = ax[1].pcolormesh(xxm, yym, zi_lin, cmap=cmap,  shading='faceted', vmin=vmin, vmax=vmax)
    ax[1].set_aspect('equal')
    ax[1].plot(xx[0, ], yy[0, ], color='k')
    ax[1].plot(xx[-1, ], yy[-1, ], color='k')
    ax[1].plot(xx[:, 0], yy[:, 0], color='k')
    ax[1].plot(xx[:, -1], yy[:, -1], color='k')
    ax[1].set_xlim([min(x), max(x)])
    ax[1].set_ylim([min(y), max(y)])
    fig.colorbar(pcm, ax=ax[1], orientation='horizontal')
    ax[1].set_title('Interpolated ADCIRC {}'.format(varname), fontsize=14)
    plt.show()


def write_tif(meshdict, zi_lin, targetgrid, targetepsg, filename='test.tif'):
    """
    Construct the new TIF file and store it to disk in filename
    """
    xm0, ym0 = meshdict['uplx'], meshdict['uply']
    transform = from_origin(xm0 - targetgrid['res'] / 2, ym0 + targetgrid['res'] / 2, targetgrid['res'], targetgrid['res'])
    utilities.log.info('TIF transform {}'.format(transform))
    # print(transform)
    nx = targetgrid['nx']
    ny = targetgrid['ny']
    crs = targetepsg
    md = {'crs': crs,
          'driver': 'GTiff',
          'height': ny,
          'width': nx,
          'count': 1,
          'dtype': zi_lin.dtype,
          'nodata': -99999,
          'transform': transform}
    # output a geo-referenced tiff
    dst = rio.open(filename, 'w', **md)
    try:
        dst.write(zi_lin, 1)
        utilities.log.info('Wrote TIF file to {}'.format(filename))
    except:
        utilities.log.error('Failed to write TIF file to {}'.format(filename))
    dst.close()

def write_png(filenametif='test.tif', filenamepng='test.png'):
    """
    Construct the new PNG file  from the tif file
    Translate will automatically create and save the new file,
    """
    from osgeo import gdal
    scale = '-scale min_val max_val'
    options_list = [
        '-ot Byte',
        '-of PNG',
        scale
    ] 
    options_string = " ".join(options_list)
    gdal.Translate(filenamepng,
           filenametif,
           options=options_string)
    utilities.log.info('Storing a PNG with the name {}'.format(filenamepng))

def plot_tif(filename='test.tif'):
    """
    Read TIF file that has been previously generated
    """
    dataset = rio.open(filename)
    band1 = dataset.read(1, masked=True)
    show(band1, cmap='jet')
    msk = dataset.read_masks(1)

def plot_png(filename='test.png'):
    """
    Read TIF file that has been previously generated
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.show()


#################################################################
## Start doing some work

# Do we need varname as in input argument ?
# Yes we do make it an enumeration of three possible things.
# Want to be able to simply input a URL
# urlinput='http://tds.renci.org:8080/thredds//dodsC/2020/nam/2020042912/hsofs/hatteras.renci.org/ncfs-dev-hsofs-nam-master/namforecast/maxele.63.nc'
#

def main(args):
    """
    Prototype script to construct geotiff files from the ADCIRC triangular grid
    """
    utilities.log.info(args)
    experimentTag = args.experiment_name
    filename = args.filename
    png_filename = args.png_filename
    varname = args.varname
    showInterpolatedPlot = args.showInterpolatedPlot
    showRasterizedPlot = args.showRasterizedPlot
    showPNGPlot = args.showPNGPlot
    
    showGDALPlot = False # Tells GDAL to load the tif and display it
    # Add in option to simply upload a url

    if not checkEnumuation(varname):
        utilities.info.error('Incorrect varname input {}'.format(varname))

    utilities.log.info('Start ADCIRC2Geotiff')
    main_config = utilities.load_config()

    if args.urljson is not None:
        setGetURL = False
        if not os.path.exists(args.urljson):
            utilities.log.error('urljson file not found.')
            sys.exit(1)
        utilities.log.info('Read data from supplied urljson')
        urls = utilities.read_json_file(args.urljson)
        dstr = '00'  # Need to fake these if you input a urljson
        cyc = '00'
    elif args.url != None:
        # If here we still need to build a dict for ADCIRC
        url = args.url
        dte='placeHolder' # The times will be determined from the real data
        urls={dte:url}
        dstr = '00'  # Need to fake these if you input a url
        cyc = '00'
        utilities.log.info('Explicit URL provided {}'.format(urls))
    else:
        utilities.log.error('No Proper URL specified')
    #else:
    #    utilities.log.info('Executing the getURL process')
    #    url, dstr, cyc = construct_url(varname)
    #    urls = {'dstr': url}

    # if not validate_url(urls):
    #     utilities.log.info('URL is invalid {}'.format(url))

    # Now construct filename destination using the dstr, cyc data

    iometadata = '_'.join([dstr, cyc])
    utilities.log.info('Attempt building dir name: {}, {}, {}'.format(
                 main_config['DEFAULT']['RDIR'], iometadata, experimentTag))

    if experimentTag is None:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='APSVIZ_'+iometadata)
    else:
        rootdir = utilities.fetchBasedir(main_config['DEFAULT']['RDIR'], basedirExtra='APSVIZ_'+experimentTag+'_'+iometadata)

#Build final pieces for the subsequent plots
#    if args.urljson is not None:
#        # my_dict.keys()[0]
#        url = list(urls.values())[0]

    targetgrid, targetepsg = read_inter_grid_yaml()
    # targetgrid, targetepsg = default_inter_grid() # A builtin option but same as yaml example
    # use the first url to get the ADCIRC grid parts, assumes all urls point to the same
    # ADCIRC grid...
    url = list(urls.values())[0]
    
    t0 = time.time()
    tri = extract_grid(url, targetepsg)
    utilities.log.info('Building ADCIRC grid took {} secs'.format(time.time()-t0))
    t0 = time.time()
    meshdict = compute_geotiff_grid(targetgrid, targetepsg)
    utilities.log.info('compute_geotiff_grid took {} secs'.format(time.time() - t0))
    xxm, yym = meshdict['xxm'], meshdict['yym']

    orig_filename = filename 
    orig_png_filename = png_filename 
    for utime, url in urls.items():
        filename = '_'.join([utime,orig_filename])
        png_filename = '_'.join([utime,orig_png_filename])
        filename = '/'.join([rootdir,filename])
        png_filename = '/'.join([rootdir,png_filename]) 
        utilities.log.info('Using tif outputfilename of {}'.format(filename))
        utilities.log.info('Using png outputfilename of {}'.format(png_filename))
        #fname = "{}.tif".format(utime)

        nc = netCDF4.Dataset(url)
        advardict = get_adcirc_slice(nc, varname)
        vmin = np.nanmin(advardict['data'])
        vmax = np.nanmax(advardict['data'])
        utilities.log.info('Min/Max in ADCIRC Slice: {}/{}'.format(vmin, vmax))

        # construct the interpolator
        t0 = time.time()
        interp_lin = Tri.LinearTriInterpolator(tri, advardict['data'])
        utilities.log.info('Finished linearInterpolator in {} secs'.format(time.time()-t0))

        # zi_lin is the interpolated data that form the rasterized data down below
        zi_lin = interp_lin(xxm, yym)
        utilities.log.debug('zi_lin {}'.format(zi_lin))

        t0 = time.time()
        write_tif(meshdict, zi_lin, targetgrid, targetepsg, filename)
        utilities.log.info('compute_mesh took {} secs'.format(time.time()-t0))

        t0 = time.time()
        write_png(filename, png_filename)
        utilities.log.info('write_png took {} secs'.format(time.time()-t0))

    if showInterpolatedPlot:
        plot_triangular_vs_interpolated(meshdict, varname, tri, zi_lin, advardict)

    if showRasterizedPlot:
        plot_tif(filename)

    if showPNGPlot:
        plot_png(png_filename)

    if showGDALPlot:
        # Can we reread the file using GDAL ?
        from osgeo import gdal
        ds = gdal.Open(filename).ReadAsArray()
        im = plt.imshow(ds)
        plt.show()

    utilities.log.info('Finished') 

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', action='store', dest='experiment_name', default=None,
                        help='Highlevel Experiment-tag value')
    parser.add_argument('--tif_filename', action='store', dest='filename', default='test.tif',
                        help='String: tif output file name will be prepended by new path. Must include extension')
    parser.add_argument('--png_filename', action='store', dest='png_filename', default='test.png',
                        help='String: png output file name will be prepended by new path. Must include extension')
    parser.add_argument('--showInterpolatedPlot', action='store_true', 
                        help='Boolean: Display the comparison of Trangular and interpolated plots')
    parser.add_argument('--showRasterizedPlot', action='store_true',
                        help='Boolean: Display the generated and saved tif plot')
    parser.add_argument('--showPNGPlot', action='store_true', 
                        help='Boolean: Display the generated and saved png plot')
    parser.add_argument('--varname', action='store', dest='varname', default='zeta_max',
                        help='String: zeta_max, vel_max, or inun_max')
    parser.add_argument('--urljson', action='store', dest='urljson', default=None,
                        help='String: Filename with a json of urls to loop over.')
    parser.add_argument('--url', action='store', dest='url', default=None,
                        help='String: url.')
    args = parser.parse_args()
    sys.exit(main(args))
