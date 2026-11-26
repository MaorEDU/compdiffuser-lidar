#!/bin/bash

##
## Env Install: Step 3
##

source ~/.bashrc
source activate compdfu_ogb_release

## Install the customized OGBench env by CompDiffuser
## Clone ogbench_cpdfu_release as a sibling directory of comp_diffuser_release, then run this.
pip install -e '../ogbench_cpdfu_release' --no-deps

## pin mujoco and setuptools versions
pip install mujoco==3.2.6
pip install setuptools==65.5.0
