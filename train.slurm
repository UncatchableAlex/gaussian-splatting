#!/bin/bash

# job name
#SBATCH --job-name=futhark_3dgs

# output logs
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Resource requests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=3
#SBATCH --exclusive
#SBATCH --exclude=hendrixgpu18fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl


# see what kinda device we're dealing with here
nvidia-smi

# start training
/home/mjk711/.conda/envs/gaussian_splatting/bin/python train.py \
 --futhark \
 --convert_SHs_python \
 --iterations 20000 \
 --start_checkpoint /home/mjk711/gaussian-splatting/output/071d7ae9-f/chkpnt5000.pth\
 --checkpoint_iterations 5000 10000 15000 20000 \
 --save_iterations 5000 10000 15000 20000 \
 --test_iterations 200 7000 20000 \
 -s ~/gaussian-splatting/tandt/train #\
#  & \
#  /home/mjk711/.conda/envs/gaussian_splatting/bin/python training_client_viewer.py \
#  --mode video \
#  --frames 15000 \
#  --stride 5 \
#  --orbits 5
