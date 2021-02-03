#!/usr/bin/env python

##
## Grab the META data PKL generated from GetObsStations: (metapkl)
## Grab the Merged ADC,OBS,ERR time series data CSV computed by (mergedf) 
## Specify list up to four stations for comparison
##
import os,sys
import numpy as np
import pandas as pd
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns
from utilities.utilities import utilities

## Some attempt at managing colors.
## For now do not change the order of these

colordict=dict()
colordict['forecast']='b'
colordict['nowcast']='gray'
colordict['adc']='b'
colordict['obs']='g'
colordict['err']='r'
colordict['misc']='o'

dashdict=dict()
dashdict['forecast']=(3,1)
dashdict['nowcast']=(3,1)
dashdict['adc']=(1,0)
dashdict['obs']=(1,0)
dashdict['err']=(1,0)
dashdict['misc']=(3,2)

class stationPlotter(object):
    """
    """
    def __init__(self, files=None, rootdir='None', iosubdir='PNGs', metadata=''):
        self.files = files
        self.rootdir=rootdir
        if self.rootdir == None:
            utilities.log.error("No rootdir specified on init {}".format(self.rootdir))
            sys.exit(1)
        self.iosubdir=iosubdir
        self.iometadata=metadata
        utilities.log.info('Invoke station-level PNG generation')

    def getBounds(self,df):
        """
        Looks into every variables and find their global min and global max
        I want a symmetric-about-zero y axis
        """
        ymin = abs(df.min().min())
        ymax = abs(df.max().max())
        val = ymax if ymax > ymin else ymin
        ymin = -math.ceil(val)
        ymax = -ymin
        return ymin, ymax

    def addPlot(self, fig, station, stationid, df_concat, variables):
        ymin, ymax = self.getBounds(df_concat)
        colorlist, dashlist = self.varToStyle(variables)
        nCols = len(variables)
        sns.set_style('darkgrid')
        ax=sns.lineplot(data=df_concat, palette=colorlist, dashes=dashlist)
        ax.legend(loc = 4,fontsize = 10)
        ax.set_ylabel(r'$\Delta$ WL (m) versus MSL')
        ax.set_ylim([ymin, ymax])
        ax.get_xaxis().set_visible(True)
        ax.set(xlabel=None)
        ax.set_title('Time range {} to {}'.format( df_concat.index.min(),df_concat.index.max()),fontdict={'fontsize': 10, 'fontweight': 'medium'})
        ax.grid(linestyle='-', linewidth='0.5', color='gray')
        plt.setp(ax.get_xticklabels(), rotation = 15)
        fig.suptitle('Station Name={}'.format(station))

    def convertToDF(self, compDict,var):
        df=pd.DataFrame(compDict[var])
        df.set_index('TIME',inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns=[''.join([var,'WL'])]
        return df

    def makeDict(self, station, df_meta, pngfile):
        lon = df_meta.loc[int(station)]['lon']
        lat = df_meta.loc[int(station)]['lat']
        node = df_meta.loc[int(station)]['Node']
        return {'LAT':str(lat), 'LON':str(lon), 'NODE':str(node), 'FILENAME':pngfile}

## Now a string manipulator to help fiund the proper color
## Uses a greedy approach
    def varToStyle(self, varlist):
        """
        Choose a color and dash style depending on variable name
        return each as ordered lists. Dash list elements are tuples
        """
        colorlist = list()
        listcolors = list(colordict.keys())
        for var in varlist:
            testvar = var.lower() 
            for test in listcolors:
                if test in testvar:
                    colorlist.append(colordict[test])
                    break
        listdashes = list(dashdict.keys())
        dashlist=list()
        for var in varlist:
            testvar = var.lower()
            for test in listdashes:
                if test in testvar:
                    dashlist.append(dashdict[test])
                    break
        return colorlist, dashlist

    def generatePNGs(self):
        """
        """
        listDicts = list()
        for key,val in self.files.items():
            print(key)
            if key=='META':
                with open(val, 'r') as fp:
                    metaDict = json.load(fp) 
                    df_meta=pd.DataFrame(metaDict)
                    df_meta.set_index('stationid',inplace=True)
            else:
                with open(val, 'r') as fp1:
                    listDicts.append(json.load(fp1))

    # check station concordance
        print('check station lists')
        stations = listDicts[0]
        for i in range(0,len(listDicts)):
            stations = stations & listDicts[i].keys()
        utilities.log.info('Total intersected stations {}'.format(len(stations)))

        newDict = {}
        for station in stations:
            newDict[station]=listDicts[0][station]
            for dicts in range(1,len(listDicts)):
                newDict[station].update(listDicts[dicts][station])

    # Now grab the list of variables (eg ADC, OBS etc). Only need to choose a single station
    # How to handle DUPLICATE variable names?

        variables = list(newDict[station].keys()) 
        utilities.log.info('Set of variables is {}'.format(variables))

    # Need to check if any variables are all nans. if so, causes plotting problems. check all stations
    # if np.isnan(compDicct = {}

        print(' Station nan check {}'.format(station))
        for station in stations:
            for variable in variables:
                if all(np.isnan(newDict[station][variable]['WL'])):
                    utilities.log.info('Removing a station for nans {}'.format(station))
                    del newDict[station]
                    break

    # Recheck station list
        stations = list(newDict.keys())
        print('Total number of non nan-empty  stations is {}'.format(len(stations)))

        runProps = dict()
        for station in stations:
            print('start station {}'.format(station))
            plt.close()
            stationName = df_meta.loc[int(station)]['stationname']
            listdfs = list()
            for var in variables:
                listdfs.append(self.convertToDF(newDict[station],var))
            df_concat = pd.concat(listdfs,axis=1)
            new_variables = df_concat.columns.to_list()
            # A per-station plot
            fig = plt.figure()
            self.addPlot(fig, stationName, station, df_concat, new_variables)
            #self.metajsonname = utilities.getSubdirectoryFileName(self.rootdir, self.iosubdir, 'obs_wl_metadata'+self.iometadata+'.json')
            pngfile = utilities.getSubdirectoryFileName(self.rootdir, self.iosubdir, station+self.iometadata+'_WL.png')
            #pngfile='_'.join([station,'WL.png'])
            plt.savefig(pngfile)
            # Create a dict of lons,lats,nodes,filenames 
            runProps[station]=self.makeDict(station, df_meta, os.path.basename(pngfile)) # Remove full path
            ##print('{}'.format(runProps))
            #plt.show()
        dictfile='png_stations.json'
        runPropsFull=dict()
        runPropsFull['STATIONS']=runProps
        utilities.write_json_file(runPropsFull, dictfile)
        return runPropsFull
