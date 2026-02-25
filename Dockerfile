FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 🔥 FIX NVIDIA EXPIRED GPG KEY ISSUE
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list || true

# System deps
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    wget \
    curl \
 && rm -rf /var/lib/apt/lists/*

## 4. THE MAGIC BULLET: Install PyAV via Conda
RUN conda install -y -c conda-forge av

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Python deps (PIN EVERYTHING TO ERA)
RUN pip install \
    "numpy<1.20" \
    "cython<3.0" \
    cython-bbox \
    scipy==1.4.1 \
    matplotlib==3.2.2 \
    easydict \
    tqdm \
    opencv-python==4.2.0.34 \
    tensorboardX==2.0 \
    yacs==0.1.7

WORKDIR /workspace/AlphAction
COPY ./AlphactionFramework /workspace/AlphAction

ENV TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;7.0;7.5"

RUN pip install -e .

CMD ["/bin/bash"]