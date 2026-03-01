#!/bin/bash
echo "Setting up MOT Environment ..."

# 1. Clear old modules and load the right ones
module purge
module load cuda/11.5.0
module load gcc/8.2.0

# 2. Activate your environment
source activate alphaction_a100

# 3. Go straight to your project folder
cd Long-Term-Multi-Person-Action-Recognition-and-Tracking/AlphactionFramework/demo

echo "Ready to go! GPU Status:"
nvidia-smi -L