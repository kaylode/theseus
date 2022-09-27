# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
ARG BASE_IMAGE=ubuntu:18.04

# Instal basic utilities
FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gcc \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda-installs
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=11.3
ARG PYTORCH_VERSION=1.12.1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${PYTORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya

# Install classification
COPY ./ /workspace/
WORKDIR /workspace/
RUN /opt/conda/bin/python -m pip install -e .


FROM ${BASE_IMAGE} as official
SHELL ["/bin/bash", "-c"]
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /workspace
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg libsm6 libxext6 \
    libjpeg-dev \
    tmux \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=conda-installs /opt/conda /opt/conda
# copy packages installed by pip
COPY --from=conda-installs /workspace /workspace

ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

WORKDIR /workspace