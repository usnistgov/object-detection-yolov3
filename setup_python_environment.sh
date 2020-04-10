# ********************************
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: You cannot run this script, you need to walk through it manurally

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ********************************

# Install anaconda3
# https://www.anaconda.com/distribution/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
# bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n pt python=3.6
conda activate pt
conda update -n base -c defaults conda

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install numpy python-lmdb scikit-image protobuf
pip install --upgrade pycuda
#conda update --all