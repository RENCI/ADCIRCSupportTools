##############
# Docker file for the creation of the adcirc support tools.
#
# to create image: docker build -t adcirc_supp:latest .
# to push image:
#       docker tag adcirc_supp:latest renciorg/adcirc_supp:latest
#       docker push renciorg/adcirc_supp:latest
##############
FROM continuumio/miniconda3

# get some credit
LABEL maintainer="jtilson@renci.org"

# make sure the container is up-to-date
RUN apt-get update

# update conda and set the download channel
RUN conda update conda && \
    conda config --add channels conda-forge

# tell conda what the shell is
RUN conda init bash

# make d directory for the repos and go there
RUN mkdir /repo
WORKDIR /repo

# get the repos
RUN git clone https://github.com/jtilson/ADCIRCSupportTools.git
RUN git clone https://github.com/GClunies/noaa_coops.git
               
# move to the install directory
WORKDIR ADCIRCSupportTools/INSTALL

# install the conda requirements
RUN conda install --yes --file conda_requirements.txt
                                
# install the pip requirements
RUN pip install -r pip_requirements.txt
                              
# change to the noaa directory
WORKDIR /repo/noaa_coops

# run the noaa seteup
RUN python setup.py install --user
                        
# change to the pipelines directory                        
WORKDIR /repo/ADCIRCSupportTools/pipelines
                                   
# make a temporary directory for the output.
# this should eventually point a common PV
RUN mkdir temp

# set the python path
ENV PYTHONPATH=/repo/ADCIRCSupportTools

# set the location of the output directory
ENV RUNTIMEDIR=/data

##########
# at this point the container is ready to accept the launch command.
# see apsviz_job.yaml for the job that launches this container.
##########