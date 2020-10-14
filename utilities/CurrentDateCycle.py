#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class CurrentDateCycle
Brian Blanton, RENCI, 2020
"""

import os
import datetime as dt

def main(args):
    print(args)
    cdc = CurrentDateCycle(doffset=args.doffset)
    print(cdc)

class CurrentDateCycle:

    def __init__(self, hoffset=0, doffset=0):
        self.cyc = ()
        self.date = ()
        self.doffset = doffset
        self.hoffset = hoffset
        self.cdc = ()
        self.date = ()
        self.datecycle = ()
        self.set(hoffset, doffset)

    def __str__(self):
        return '%s' % self.cdc

    def set(self, hoffset=0, doffset=0):
        thisdate = dt.datetime.utcnow() + dt.timedelta(days=doffset) + dt.timedelta(hours=hoffset)
        self.cyc = 6 * int(thisdate.hour / 6)
        self.date = dt.date(thisdate.year, thisdate.month, thisdate.day)
        self.datecycle = dt.datetime(thisdate.year, thisdate.month, thisdate.day) + dt.timedelta(hours=self.cyc)
        self.cdc = '%s%s' % (self.datecycle.date().strftime("%Y%m%d"), self.cyc)

    def write(self):
        with open("currentDateCycle.txt", "w") as file:
            file.write(str(self.cdc))

    def read(self):
        if not os.path.exists("currentDateCycle.txt"):
            # print("currentDateCycle_test.txt not found.")
            self.cdc = ()
        else:
            with open("currentDateCycle.txt", "r") as file:
                temp = file.read()
                self.cdc = temp
                if not temp:
                    self.date, temp = self.cdc.split('_')
                    self.cyc = temp.split('z')[0]
                else:
                    self.date = ()
                    self.cyc = ()

#
# def currentDate(offset=0):
#     date = dt.datetime.utcnow()+ timedelta(days=offset)
#     date = date.strftime("%Y%m%d")
#     return date
#
# def currentCycle(offset=0):
#     hour = dt.datetime.utcnow()+timedelta(hours=offset)
#     hour = hour.hour
#     cyc = 6*(hour/6)
#     return cyc
#
# def GetDateCycle(self, doffset=0, hoffset=0):
#     """
#     returns the UTC date and synoptic cycle, adjusted for day and hour offsets
#     :param doffset: day offset to current time, negative is backward in time
#     :param hoffset: hour offset to current time, negative is backward in time
#     :return: date YYYY-MM-DD as type datetime.date
#              cycle current cycle (0,6,12,18)
#     """
#     current_t = dt.datetime.utcnow()
#     thisdate = current_t + dt.timedelta(days=doffset) + dt.timedelta(hours=hoffset)
#     hour = thisdate.hour
#     cyc = 6*int(hour/6)
#     thisdate = dt.datetime(thisdate.year, thisdate.month, thisdate.day) + dt.timedelta(hours=cyc)
#     return thisdate, cyc


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description=main.__doc__)
    # parser.add_argument('--cdat', default="0", help='Date to run system for.', type=str)
    # parser.add_argument('--ccyc', default="-1", help='Cycle to run system for.', type=str)
    # parser.add_argument('--hoffset', default=0, help='Hour offset for WW3 download.', type=int)
    parser.add_argument('--doffset', default=0, help='Day offset for WW3 download.', type=int)
    args = parser.parse_args()

    sys.exit(main(args))