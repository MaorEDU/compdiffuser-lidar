#!/bin/bash

##
## Env Install: Step 2
##

source ~/.bashrc
source activate compdfu_ogb_release

## install gymnasium env
pip install -e git+https://github.com/Farama-Foundation/Gymnasium.git@81b87efb9f011e975f3b646bab6b7871c522e15e#egg=gymnasium
pip install -e git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@3e42c3061aa23ce7f29ae8ec150ca6224e13ef09#egg=gymnasium_robotics

## install torch + torchvision for CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

## install xformers matching torch 2.5.0 (CUDA build)
pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121
