
for YEAR in 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019; do
     echo $YEAR
     #ls /projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/40YearEC95D/EC95D/YEARLY-$YEAR/DAILY-ec95d-RANGE8-SILL0.16-NUGGET0.001-LP24/interpolated/ADC*.csv | wc -l 
     ls /projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/ADCIRCSupportTools/reanalysis/40YearEC95D/EC95D-DA/YEARLY-$YEAR/DAILY-ec95d-RANGE8-SILL0.16-NUGGET0.001-LP24/interpolated/ADC*.csv | wc -l 
done


