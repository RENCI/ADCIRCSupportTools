FROM continuumio/miniconda3

# get some credit
LABEL maintainer="jtilson@renci.org"

RUN apt-get update

RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda create --name myenv -y python=3.7.3
    
RUN conda init bash

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN mkdir /repo
WORKDIR /repo

# commented out for now. this repo should be made public or pass credentials into the container.
#RUN git clone https://github.com/jtilson/ADCIRCSupportTools.git
#WORKDIR ADCIRCSupportTools/INSTALL
#RUN conda install --yes --file conda_requirements.txt
#run pip install -r pip_requirements.txt

# 3) Install the local noaa-coops

WORKDIR /repo

RUN git clone https://github.com/GClunies/noaa_coops.git
WORKDIR noaa_coops

