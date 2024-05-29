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

Clone our code
git clone https://github.com/Llili314/IRHrNet：

Remember to activate your virtual enviroment before running our code：
conda activate IRHrNet

# Replicate our method on heart rate estimation from facial videos by modifying or running the following scripts
net_dual.py # CET & MDEF combination module
fusion_strategy.py # defination of MDEF
args_fusion_dual.py # experimental parameters setting
utils.py # refers to utility functions
train_rppg.sh # slurm train shell
train_rppg.py # train CET and MDEF modules to extract the rPPG signals 
train_hr.py # train FEE module cascaded with CET and MDEF modules  
test.py  # test rppg and hr; In order to test the trained model, the model can be downloaded from the Baidu disk (Link: https://pan.baidu.com/s/1JJCwS0WrJ_55hM63Unni5g, Code: 52fz)
