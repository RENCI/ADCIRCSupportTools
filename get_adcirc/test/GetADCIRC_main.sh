
# OPTIONS for manual execution
# Input times
python ../GetADCIRC.py --ignore_pkl --timein='2020-09-17 12:00' --timeout='2020-09-21 18:00'

python  ../GetADCIRC.py --ignore_pkl --timeout='2020-09-21 18:00' --doffset=-2

# input json of url(s)
python ../GetADCIRC.py --ignore_pkl --urljson data1.json

# input json of url(s)
python ../GetADCIRC.py --grid 'hsofs' --ignore_pkl --urljson data1.json


# Run a different grid
python ../GetADCIRC.py --ignore_pkl --grid 'ec95d' --urljson ec95d_data1.json 

# Run a different greid forecast
python ../GetADCIRC.py --ignore_pkl --grid 'ec95d' --urljson ec95d_data1_forecast.json

# MODIFY defauilt filenames and subdirs

python  ../GetADCIRC.py --ignore_pkl --timeout='2020-09-21 18:00' --doffset=-2 --iometadata='_myfile' --iosubdir='MYDIR'

# Call a hurricane
python  ../GetADCIRC.py --ignore_pkl --grid 'ec95d' --urljson data1_hurricaneFlorence_forecast.json

# Call a hurricane
python  ../GetADCIRC.py --ignore_pkl --writeJson --grid 'ec95d' --urljson data1_ec95d.json 

# Call LA new data
python  ../GetADCIRC.py --ignore_pkl --writeJson --grid "LA_v20a-WithUpperAtch_chk" --urljson data1_LAtest.json
