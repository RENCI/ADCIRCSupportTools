# A typical invocation

# Single kriging
python performance_CV.py --errorfile '/home/jtilson/ADCIRCSupportTools/ADCIRC/stationSummaryAves_manualtest.csv' --gridjsonfile '/home/jtilson/ADCIRCSupportTools/get_adcirc/ADCIRC/adc_coord.json' --clampfile '/home/jtilson/ADCIRCSupportTools/config/clamp_list_hsofs.dat' --subdir_name 'TESTINT' --iometadata 'IOMETA'

# Optimize and grab clampfile from config

python performance_CV.py --errorfile '/home/jtilson/ADCIRCSupportTools/ADCIRC/stationSummaryAves_manualtest.csv' --gridjsonfile '/home/jtilson/ADCIRCSupportTools/get_adcirc/ADCIRC/adc_coord.json' --clampfile '/home/jtilson/ADCIRCSupportTools/config/clamp_list_hsofs.dat' --subdir_name 'TESTINT' --iometadata 'IOMETA' --cv_kriging
