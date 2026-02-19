# 1. The Bulletproof Base Image: Official PyTorch 1.10.0 + CUDA 11.3 + cuDNN 8
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 2. CRITICAL FIX: Delete expired NVIDIA GPG keys so apt-get update doesn't crash
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# 3. Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 4. THE MAGIC BULLET: Install PyAV via Conda
RUN conda install -y -c conda-forge av

# 5. Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# 6. Install the rest of the Python Dependencies via pip
RUN pip install \
    "numpy<1.24" \
    "cython<3.0" \
    cython-bbox \
    scipy \
    matplotlib \
    easydict \
    tqdm \
    opencv-python \
    tensorboardX \
    yacs

WORKDIR /workspace/AlphAction

# 7. Copy code last to maximize Docker build cache efficiency
COPY ./AlphactionFramework /workspace/AlphAction

# 8. Define target architectures explicitly to prevent PyTorch auto-detect crashes
# This covers everything from older local GPUs up to modern HPC cluster hardware
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# 9. Build the AlphAction C++/CUDA extensions
RUN pip install -e .

ENV PYTHONPATH="/workspace/AlphAction:${PYTHONPATH}"

CMD ["/bin/bash"]