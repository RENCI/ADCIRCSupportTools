#!/usr/bin/env python

# Class to manage the construction of the ADCIRC-OBSERVATONS error fields
# Mostly this class manages filenames as computing the error is easy. What is hard
# is we wish to average error files over a user-specified number of cycles.
#
# Check the station df_final error measures. If z-score > 3 remove them 

import os, sys
import time as tm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utilities.utilities import utilities
from scipy import stats
from compute_error_field.computeErrorField import computeErrorField


import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def main(args):
    # print (args)
    meta = args.obsmeta
    obsf = args.obsdata
    adcf = args.adcdata
    missingFiles=True
    if (meta!=None) and (obsf!=None) and (adcf!=None):
        missingFiles=False
    timedata = list()
    for itimes in range(0,10):
        t0 = tm.time()

        # We can change the class to use the load_config() eliminating the need for this yaml ref.
        config = utilities.load_config()
        extraExpDir='TimeComputeError'
        rootdir=utilities.fetchBasedir(config['DEFAULT']['RDIR'],basedirExtra=extraExpDir+str(itimes))
        if missingFiles:
            dir = '/home/jtilson/ADCIRCSupportTools/ADCIRC'
            utilities.log.info('1 or more inputs files missing Try instead to find in dir {}'.format(dir))
            meta='/'.join([dir,'obs_wl_metadata.pkl'])
            obsf='/'.join([dir,'obs_wl_smoothed.pkl'])
            adcf='/'.join([dir,'adc_wl.pkl'])
        print('Run computeErrorField')
        cmp = computeErrorField(obsf, adcf, meta, rootdir=rootdir, aveper=4)
        testSubdir='testDir'
        cmp.executePipeline(metadata = 'timing',subdir=testSubdir)
        errf, finalf, cyclef, metaf, mergedf = cmp._fetchOutputFilenames()
        print('output files '+errf+' '+finalf+' '+cyclef+' '+metaf+' '+mergedf)
        timedata.append( tm.time()-t0 )
    m, mmh,mph = mean_confidence_interval(timedata)
    print('Total times: Mean {} 95% CI range {}-{}'.format(m, mmh, mph))

if __name__ == '__main__':
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument('--obsmeta', action='store', dest='obsmeta',default=None, help='FQFN to obs_wl_metadata.pkl', type=str)
    parser.add_argument('--obsdata', action='store', dest='obsdata',default=None, help='FQFN to obs_wl_smoothed.pkl', type=str)
    parser.add_argument('--adcdata', action='store', dest='adcdata',default=None, help='FQFN to adc_wl.pkl', type=str)
    args = parser.parse_args()
    sys.exit(main(args))

