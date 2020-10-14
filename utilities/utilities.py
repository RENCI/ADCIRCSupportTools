#!/usr/bin/env python

#############################################################
#
# Retain from original utilities, only file IO, URL processing, and logging methods
# Add timing calls
# That can be used by any of the ADCIRC support tools
#
# RENCI 2020
#############################################################

import datetime as dt
import pandas as pd
import sys,os
import yaml
import logging
import json
from argparse import ArgumentParser

LOGGER = None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Utilities:
    """

    """
    def __init__(self):
        """
        Initialize the Utilities class, set up logging
        """
        global LOGGER
        self.config = self.load_config()

        if LOGGER is None and self.config["DEFAULT"]["LOGGING"]:
            log = self.initialize_logging()
            LOGGER = log
        self.log = LOGGER

#############################################################
# Logging

    def initialize_logging(self):
        """
        Initialize project logging
        """
        # logger = logging.getLogger(__name__)
        logger = logging.getLogger("adda_services")
        log_level = self.config["DEFAULT"].get('LOGLEVEL', 'DEBUG')
        # log_level = getattr(logging, self.config["DEFAULT"].get('LOGLEVEL', 'DEBUG'))
        logger.setLevel(log_level)

        # LogFile = self.config['LOG_FILE']
        # LogFile = '{}.{}.log'.format(thisDomain, currentdatecycle.cdc)
        LogFile = 'log'
        # print('Use a log filename of '+LogFile)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(funcName)s : %(module)s : %(name)s : %(message)s ')
        dirname = os.path.dirname(LogFile)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        file_handler = logging.FileHandler(LogFile, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # logging stream
        # formatter = logging.Formatter('%(asctime)s - %(process)d - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s')
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        return logger

#############################################################
# YAML
    def load_config(self, yaml_file=os.path.join(os.path.dirname(__file__), '../config', 'main.yml')):
        #yaml_file = os.path.join(os.path.dirname(__file__), "../config/", "main.yml")
        if not os.path.exists(yaml_file):
            raise IOError("Failed to load yaml config file {}".format(yaml_file))
        with open(yaml_file, 'r') as stream:
            config = yaml.safe_load(stream)
            print('Opened yaml file {}'.format(yaml_file,))
        self.config = config
        return config

    def print_dict(self, t, s):
        if not isinstance(t, dict) and not isinstance(t, list):
            # pass
            if not isinstance(t, float):
                print("\t" * s + str(t))
        else:
            for key in t:
                if not isinstance(t, float) and not isinstance(t, list) \
                        and not isinstance(t, int) and not isinstance(t, unicode):
                    print("\t" * s + str(key))
                if not isinstance(t, list):
                    self.print_dict(t[key], s + 1)

    def serializeMe(self, o):
        if isinstance(o, dt.datetime):
            return o.__str__()

    def readConfigYml(self, yamlfilename):
        if not os.path.exists(yamlfilename):
            raise IOError("Failed to find config file %s" % yamlfilename)
        # config_file = EnvYAML(yamlfilename)
        # print(config_file['ADDAHOME'])
        with open(yamlfilename, 'r') as stream:
            config_file = yaml.safe_load(stream)
        return config_file

#############################################################
# IO uses the base YAML config to do its work

    def fetchBasedir(self, inconfig, basedirExtra='None'):
        try:
            rundir = os.environ[inconfig.replace('$', '')]  # Yaml call to be subsequently removed
        except:
            print('Chosen basedir invalid: '+str(inconfig['DEFAULT']['RDIR']))
            print('reset to CWD')
            rundir = os.getcwd()
        if basedirExtra is not None:
            rundir = rundir+'/'+basedirExtra
            if not os.path.exists(rundir):
                #print("Create high level Cycle dir space at "+rundir)
                try:
                    #os.mkdir(rundir)
                    os.makedirs(rundir)
                except OSError:
                    sys.exit("Creation of the high level run directory %s failed" % rundir)
        return rundir

    def getSubdirectoryFileName(self, basedir, subdir, fname ):
        """Check and existance of and construct filenames for 
        storing the image data. basedir/subdir/filename 
        subdir is created as needed.
        """
        # print(basedir)
        if not os.path.exists(basedir):
            sys.exit("Basepath for dumping files non existent "+str(basedir))
        fulldir = os.path.join(basedir, subdir)
        if not os.path.exists(fulldir):
            #print("Create datastation dir space at "+fulldir)
            try:
                os.mkdir(fulldir)
            except OSError:
                sys.exit("Creation of the directory %s failed" % fulldir)
            #else:
            #    print("Successfully created the directory %s " % fulldir)
        return os.path.join(fulldir, fname)

    def writePickle(self, df, rootdir='.',subdir='obspkl',fileroot='filename',iometadata='Nometadata'):
        """ 
        Returns full filename for capture
        """
        newfilename=None
        try:
            mdir = rootdir
            newfilename = self.getSubdirectoryFileName(mdir, subdir, fileroot+iometadata+'.pkl')
            df.to_pickle(newfilename)
            print('Wrote pickle file {}'.format(newfilename))
        except IOError:
            raise IOerror("Failed to write PKL file %s" % (newfilename))
        return newfilename

    def writeCsv(self, df, rootdir='.',subdir='obspkl',fileroot='filename',iometadata='Nometadata'):
        """
        Write out current self.excludeList to disk as a csv
        output to rootdir/obspkl/.
        """
        newfilename=None
        try:
            mdir = rootdir
            newfilename = self.getSubdirectoryFileName(mdir, subdir, fileroot+iometadata+'.csv')
            df.to_csv(newfilename)
            print('Wrote CSV file {}'.format(newfilename))
        except IOError:
            raise IOerror("Failed to write file %s" % (newfilename))
        return newfilename

    def read_json_file(self, filepath):
        # Read data from JSON file specified by full path
        data = {}
        try:
            with open(filepath, 'r') as fp:
                data = json.load(fp)
        except FileNotFoundError:
            raise FileNotFoundError("Failed to read file %s" % (filepath))               
        return data

    def write_json_file(self, data, filepath):
        # write data from JSON file specified by full path
        try:
            with open(filepath, 'w') as fp:
                json.dump(data, fp)
        except IOError:
            raise IOerror("Failed to write JSON file %s" % (filepath))

# Below here to be deleted 

    def reg_grid_params(self):
        # load list of reg grid params
        return self.config["REGRID"]["RECT"]

    def get_clamp_list(self):
        # load list of zero-clamp points
        ffile = os.path.join(os.path.dirname(__file__), "../config/", self.config['DEFAULT']['ClampList'])
        if not os.path.exists(ffile):
            raise IOError("Failed to load clamplist file")
        df = pd.read_csv(ffile)
        return df

    def get_station_list(self):
        sfile = os.path.join(os.path.dirname(__file__), "../config/", self.config["DEFAULT"]["StationFile"])
        if not os.path.exists(sfile):
            raise IOError("Failed to load station list file")
        df = pd.read_csv(sfile, header=[0, 1], index_col=0)
        return df

    def get_adcirc_nodes(self):
        sfile = os.path.join(os.path.dirname(__file__), "../config/", self.config["ADCIRC"]["NodeList"])
        if not os.path.exists(sfile):
            raise IOError("Failed to load ADCIRC node list file")
        # columns = ["nodenumber", "lon", "lat", "z"]
        df = pd.read_csv(sfile, header=0, index_col=0,  sep='\s+', engine='python')  # , names=columns)
        return df

    def print_dict(self, t, s):
        if not isinstance(t, dict) and not isinstance(t, list):
            # pass
            if not isinstance(t, float):
                print("\t" * s + str(t))
        else:
            for key in t:
                if not isinstance(t, float) and not isinstance(t, list) \
                        and not isinstance(t, int) and not isinstance(t, unicode):
                    print("\t" * s + str(key))
                if not isinstance(t, list):
                    self.print_dict(t[key], s + 1)


#############################################################

utilities = Utilities()