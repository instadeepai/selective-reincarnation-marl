FROM mambaorg/micromamba:0.24.0 as conda

# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git pip swig cmake libpython3.8 && \
    rm -rf /var/lib/apt/lists/*

## Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --chown=$MAMBA_USER:$MAMBA_USER docker_config.yaml /tmp/environment.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM ubuntu:22.04 as test-image

# Let's have git installed in the container.
RUN apt update
RUN apt install -y git curl

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/sr-marl/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Africa/Johannesburg
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/sr-marl/lib

COPY . $APP_FOLDER
RUN pip install --user -e ./

FROM test-image as run-image

WORKDIR /home/app/rmarl
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.8 \
        python3.8-dev && \
    apt-get install -y python3-pip \
        swig \
        python3-opencv \
        curl \
        git \
        ffmpeg \
        unzip \
        wget \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf

RUN python -V
ADD requirements.txt .
RUN pip install --upgrade protobuf==3.20.*
RUN pip install --upgrade setuptools wheel
RUN pip install -r requirements.txt

# INSTALL environment
ADD install_mamujoco.sh .
ADD mamujoco_requirements.txt .
RUN bash install_mamujoco.sh

ENV PYTHONPATH "${PYTHONPATH}:${folder}"
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

EXPOSE 6006
