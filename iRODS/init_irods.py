#!/usr/bin/env python

# Simply perform the iinit and some simple checks on the iRODS account
# That is made available from the provided irods_environment.json file
# The provided password must correspond to the user in the irods json file
# This only needs to be run ONCE to build a local encrypted passwd file 
#

import os, sys
import shutil
import time as tm
from utilities.utilities import utilities as utilities

class iRODS:
    """
    Basic connections to an existing iRODS vauly assuming an irods_connection.json has already 
    been put into place
    """
    def __init__(self):
        utilities.log.info('Attempt to initialize iRODS')
        passwd = os.getenv('IRODS_PASSWD')
        try:
            cmd='iinit '+passwd
            os.system(cmd)
        except KeyError as e:
        utilities.log.error('Error initializing iRODS: Might as well abort')
        utilities.log.error(e)
        sys.exit(1)

    def testAccess(self):
        utilities.log.info('Attempt to test simple access to the current iRODS')
        try:
            utilities.log.info(' Result of an ils {}'.format(os.system('ils')))
        except KeyError as e:
            utilities.log.error('Failed ils testing')
            utilities.log.error(e)
            sys.exit(1)

    def getEnv(self):
        utilities.log.info('iRODS current env. Must have iinit first')
        try:
            utilities.log.info(' Result of an ils {}'.format(os.system('ienv')))
        except KeyError as e:
            utilities.log.error('Failed ienv reporting')
            utilities.log.error(e)
            sys.exit(1)

def main(args):
    """
    Initializes iRODS and attemps some simple tests to check basic sanity
    """
    utilities.log.info("Try to initialize and check iRODS")
    irods = iRODS()
    irods.testAccess()
    irods.getEnv()
    utilities.log.info('iRODS initialization successfully')

if __name__ == '__main__':
    sys.exit(main(args))
