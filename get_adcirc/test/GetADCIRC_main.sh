
# OPTIONS for manual execution
# Input times
python ../GetADCIRC.py --ignore_pkl --timein='2020-09-17 12' --timeout='2020-09-21 18'

python  ../GetADCIRC.py --ignore_pkl --timeout='2020-09-21 18' --doffset=-2

# input json of url(s)

python ../GetADCIRC.py --ignore_pkl --urljson data1.json

# MODIFY defauilt filenames and subdirs

python  ../GetADCIRC.py --ignore_pkl --timeout='2020-09-21 18' --doffset=-2 --iometadata='_myfile' --iosubdir='MYDIR'

