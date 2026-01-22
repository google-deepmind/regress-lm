
ARG PYTORCH="2.5.1"
ARG CUDA="12.1"
ARG CUDNN="9"

FROM sfoucherq/mamba-ssm:latest
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS base
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN conda update conda && conda install pip && conda clean -afy
WORKDIR /mindsom 
COPY environment-dev.yml ./
RUN conda env update -f environment-dev.yml -n base && conda clean -afy

FROM base AS dev
# Install make for the docs build
# solves a weired problem with NVIDIA with https://github.com/NVIDIA/nvidia-container-toolkit/issues/258#issuecomment-1903945418
RUN \
    # Update nvidia GPG key
    rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

RUN apt-get update  \
    && apt-get install -y --no-install-recommends python3-pyqt5 python3-pyqt5.qtwebengine unzip git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY . ./
# pre-commit run --all-files fails w/o this line
RUN git init
RUN pip install -e .
