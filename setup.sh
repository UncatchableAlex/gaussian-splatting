#!/usr/bin/env bash
set -e  # exit on any error

# create and activate the conda environment
conda env create -f environment.yml
conda activate gtest

# 1. load CUDA — this will inject the stubs path
module load cuda/11.6

# 2. strip cuda stubs that module load just added. This step is necessary to satisfy pip
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v stubs | tr '\n' ':')

# 3. install pip packages
pip install \
    submodules/diff-gaussian-rasterization \
    submodules/simple-knn \
    submodules/fused-ssim \
    opencv-python \
    joblib \
    submodules/futhark-server-python \
    matplotlib \
    numpy \
    -e submodules/futhark-3dgs

# 4. build the Futhark rasterizer. As far as I know, this versions of futhark past futhark-0.25.32 don't work
cd submodules/futhark-3dgs/futhark-rasterizer
futhark pkg sync
futhark cuda --server rasterizer.fut