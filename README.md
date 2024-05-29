# IRHrNet
"This repo is used for the paper "Illumination variation-resistant network for heart rate measurement by exploring RGB and MSR spaces", which is submitted in IEEE TIM."

# Requirements
pytorch=1.11.0 +
python=3.9 +
torchvision
numpy
opencv-python
anaconda (suggeted)

# Configuration
Create your virtual environment using anaconda：
conda create -n IRHrNet python=3.9

Activate your virtual environment：
conda activate IRHrNet

Clone our code：
git clone https://github.com/Llili314/IRHrNet

Remember to activate your virtual enviroment before running our code：
conda activate IRHrNet

Replicate our method on heart rate estimation from facial videos by modifying or running the following scripts：

RMformer.py # Defined module

args_fusion_128.py # experimental parameters setting

utils.py # refers to utility functions

train_128.sh # slurm train shell

train_128.py # train modules to extract the rPPG signals 

test_hr_128.py  # test rppg and hr;
