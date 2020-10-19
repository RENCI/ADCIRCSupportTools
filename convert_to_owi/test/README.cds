#####################################################
# cds_request.py
#
# This method fetches and stores MONTHLY data from the cds site
# Basic run scenarios for the cds_request method
#
# 0) Log onto https://cds.climate.copernicus.eu/ and create an account.
# 1) On your local machine ( say HT), in ~, build the file .cdsapirc. Add the the Key and User ID information 
# as specified when you create the account. A FAKE example is:
#
# (base) [jtilson@ht3 ~]$ cat .cdsapirc 
# url: https://cds.climate.copernicus.eu/api/v2
# key: 11962:067d20d2-4242-42fb-ab42-f602e42d09ad
#
# 2) Set up the ../config/cds.yml file. Generally, you only need to do this once as the typical entries a 
# user may change (year,month) can be done on the application CLI.
# To run the worldgrid for all the years.
# NOTE year and months can be overridden at the command line.  NOTE no area term is included here but could be;
# THe AREA is optional BUT, changing this can have downstream consequences.
# Namely, by defalt CDS returns the data in long of DegEast. Adding the area forces it to be [-180,180]
#
CDS: &cds
  product_type: 'reanalysis'
#   area: [50, -180, 5, 180]
  month: [ '01','02','03','04',
           '05','06','07','08',
           '09','10','11','12']
  day: ['01', '02', '03',
         '04', '05', '06',
         '07', '08', '09',
         '10', '11', '12',
         '13', '14', '15',
         '16', '17', '18',
         '19', '20', '21',
         '22', '23', '24',
         '25', '26', '27',
         '28', '29', '30',
         '31']
  time: ['00:00', '01:00', '02:00',
         '03:00', '04:00', '05:00',
         '06:00', '07:00', '08:00',
         '09:00', '10:00', '11:00',
         '12:00', '13:00', '14:00',
         '15:00', '16:00', '17:00',
         '18:00', '19:00', '20:00',
         '21:00', '22:00', '23:00']
  year: '2018'
  variable: ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']
  format: 'netcdf'

#2) Ensure you have a proper python env installed. For example on HT, I use:
# For example: 
conda create -n cdsdenv xarray basemap matplotlib pandas matplotlib netCDF4 pyyaml
conda activate cdsdenv
pip install cdsapi
pip install -r requirements.txt

#3) Initialize the env 

cd /projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools
conda activate cdsdenv
export PYTHONPATH=/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools:$PYTHONPATH
export RUNTIMEDIR="."

# 3) Examples of running the code:
# Several ways to invoke the code are possible and depend on a combination of YML and CLI values.

# This will fetch the netCDF file for the specified year,month and create the file $RUNTIMEDIR/"ERA5"/year/month.nc
# The "ERA5" is added by the application itself.

# 3a) 
export year='2018'
export month='01'
python -u ../cds_request.py --year $year --month $month

# 3b) simply grab the year an month data from the YML.  If multiple months in the YAML
# Then the resulting file will include those multiple months

python -u $dir/cds_request.py

# 4) A typical run case using Slurm

import os, sys
def build_slurm(year,month):
    slurm = list()
    slurm.append('#!/bin/sh')
    slurm.append('#SBATCH -t 128:00:00')
    slurm.append('#SBATCH -p batch')
    slurm.append('#SBATCH -N 1')
    slurm.append('#SBATCH -n 2')
    slurm.append('#SBATCH -J CDS'+year)
    slurm.append('#SBATCH --mem-per-cpu 32000')
    slurm.append('echo "Begin the CDS query" ')
    slurm.append('export PYTHONPATH=/home/jtilson/ADCIRCSupportTools:$PYTHONPATH')
    slurm.append('dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/convert_to_owi"')
    slurm.append('python -u $dir/cds_request.py --year "'+year+'" --month "'+month+'"')
    shName = '_'.join([year,'runSlurm.sh'])
    with open(shName, 'w') as file:
        for row in slurm:
            file.write(row+'\n')
    file.close()
    return (shName)

#yearlist = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
yearlist = ['2018']
month = '01'

for iyear in yearlist:
    year = str(iyear)
    print('Processing CDS {}'.format(year))
    slurmFilename = build_slurm(year, month)
    cmd = 'sbatch ./'+slurmFilename
    print('Launching job {} as {}'.format(year,slurmFilename))
    os.system(cmd) 


# 5) The same typical runcase using GNU parallel

export dir="/projects/sequence_analysis/vol1/prediction_work/ADCIRCSupportTools/convert_to_owi"
~bblanton/bin/parallel/parallel -j2  $dir/cds_request.py --metadata 'ERA5/global' --year {1} --month {2}  >> list  ::: `seq 2015 2015` ::: "01" "02" 

