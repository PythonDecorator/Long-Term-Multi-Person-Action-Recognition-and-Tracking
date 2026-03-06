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
conda create -n alphaction_a100 python=3.7 -y
conda activate alphaction_a100

pip install --upgrade pip setuptools wheel

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install "numpy<1.20" "cython<3.0" cython-bbox scipy==1.4.1
pip install ninja yacs matplotlib tqdm opencv-python

pip install \
    easydict \
    tensorboardX \
    
conda install -c conda-forge av

# unload any existing CUDA or GCC modules to avoid conflicts with the conda environment
module purge
module load cuda/11.5.0
module load gcc/8.2.0

git clone https://github.com/PythonDecorator/Long-Term-Multi-Person-Action-Recognition-and-Tracking.git
cd Long-Term-Multi-Person-Action-Recognition-and-Tracking/AlphactionFramework  # directory with setup.py

FORCE_CUDA=1 pip install -e .

# if any build fails make sure to remove them and before trying 
# again with the correct CUDA toolkit and GCC versions as shown in the next step
rm -rf build alphaction_a100.egg-info
conda install -c conda-forge libiconv -y

# Install ipykernel and link it to your Jupyter setup -  
# this allows you to select the "Python (alphaction_a100)" kernel in Jupyter notebooks
conda install -y ipython ipykernel
python -m ipykernel install --user --name alphaction_a100 --display-name "Python (alphaction_a100)"

# There is a start_project.sh script included in the project that sets up the environment.
sbatch start_project.sh
```

### 3️⃣ Run the Demo
Cd into the project directory and run the demo script with the appropriate paths to your video, 
config, and model weights:

```bash
cd demo

python3 demo.py \
    --video-path test_video.mp4 \
    --output-path output.mp4 \
    --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
    --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth

```

---