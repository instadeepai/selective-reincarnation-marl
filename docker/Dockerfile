FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

## WORKDIR
WORKDIR /home/app/rmarl

## ENV VARIABLES
# Ensure no installs try launch interactive screen
ENV DEBIAN_FRONTEND=noninteractive
# Ensure outputs are logged in real-time
ENV PYTHONUNBUFFERED=1
# Add offline_marl to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/home/app/rmarl"

## UPDATE & INSTALL applications
RUN apt-get -y --fix-missing update && apt-get -y upgrade
RUN apt-get install -y curl python3-pip python3-opencv swig ffmpeg git unzip wget libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
RUN pip install --upgrade protobuf==3.20.*

## COPY files to container
COPY . . 

## INSTALL offline_marl
RUN pip install -r requirements.txt

# INSTALL environment
RUN bash install_mamujoco.sh

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
