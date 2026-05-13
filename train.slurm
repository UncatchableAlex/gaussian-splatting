#!/bin/bash

# job name
#SBATCH --job-name=futhark_3dgs

# output logs
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Resource requests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=3
#SBATCH --exclude=hendrixgpu18fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl


# see what kinda device we're dealing with here
nvidia-smi

# start training
/home/mjk711/.conda/envs/gaussian_splatting/bin/python train.py \
 --convert_SHs_python \
 --iterations 30000 \
 --checkpoint_iterations 30000 \
 --save_iterations 30000 \
 --test_iterations 25000 30000 \
 -s ~/gaussian-splatting/tandt/train \
 --futhark \
 --start_checkpoint /home/mjk711/gaussian-splatting/output/_futhark20000_8k/chkpnt20000.pth
#  & \
#  /home/mjk711/.conda/envs/gaussian_splatting/bin/python training_client_viewer.py \
#  --mode video \
#  --frames 15000 \
#  --stride 5 \
#  --orbits 5
