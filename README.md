# Long-Term Multi-Person Action Recognition & Tracking (MOT Project)
![output_video-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a6b3d839-01fc-4c4b-aa3c-a4fa02cf0ca2)

This project reproduces and runs the **AlphAction** framework for long-term multi-person action recognition and tracking using GPU acceleration via Docker.
The system performs:

- ✅ Person detection (YOLO-based tracking)
- ✅ Multi-object tracking
- ✅ Temporal feature extraction
- ✅ Action classification (AVA-style action labels)
- ✅ Video rendering with bounding boxes + predicted actions

---

## 🚀 Environment (Reproducible Setup)

This project runs inside Docker with:

- Python 3.7
- PyTorch 1.4.0
- CUDA 10.1
- cuDNN 7
- Ubuntu 18.04

⚠️ **Important:** This repository depends on legacy PyTorch APIs. 
Newer versions of PyTorch (≥1.10) will break due to `Conv3D` API changes.

---

## 🐳 Docker Usage

### 1️⃣ Build the Image

```bash
docker compose build --no-cache
```

- When you build always run pip install -e . inside the project directory to update the package.
- If you want to run the demo, make sure to copy the test video and config files into the container or mount them as volumes. 
The paths in the command below assume they are located in the parent directory of the project inside the container. 
Adjust paths as needed based on your setup.


### 2️⃣ Notebook Mode (Interactive)
If you want to run the code interactively in a Jupyter notebook, like Viper, 
where docker is not used, you can set up a conda environment with the required dependencies:
link the kernel to your Jupyter setup and install the necessary packages:

```bash
conda create -n alphaction python=3.7 -y
conda activate alphaction

# Install ipykernel and link it to your Jupyter setup
conda install -y ipython ipykernel
python -m ipykernel install --user --name alphaction --display-name "Python (alphaction)"

pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html
conda install -y -c conda-forge av

pip install --upgrade pip setuptools wheel

pip install \
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
    
git clone https://github.com/PythonDecorator/Long-Term-Multi-Person-Action-Recognition-and-Tracking.git

cd Long-Term-Multi-Person-Action-Recognition-and-Tracking/AlphactionFramework  # directory with setup.py
export TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;7.0;7.5"
pip install -e .
```

If there was an error during installation it may be compatibility issues.

### 👉 `Run this:`

```bash
# Install GCC and G++ 7
conda install -y -c conda-forge gcc_linux-64=7 gxx_linux-64=7

# Install the CUDA 10.1 development toolkit (which includes the correct nvcc)
conda install -y -c conda-forge cudatoolkit-dev=10.1

# Force the system to use the Conda GCC/G++ compilers
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++

# Force PyTorch to look for CUDA inside your conda environment
export CUDA_HOME=$CONDA_PREFIX

module purge
module load cuda/11.5.0
module load gcc/8.2.0

cd AlphactionFramework
rm -rf build
rm -rf *.egg-info
pip install -e .
```

### 3️⃣ Run the Demo
Cd into the project directory and run the demo script with the appropriate paths to your video, 
config, and model weights:

```bash
python3 demo.py \
    --video-path test_video.mp4 \
    --output-path output.mp4 \
    --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
    --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth

```
