#!/bin/bash

#SBATCH --output=%j-eureka-video.out
#SBATCH --job-name=eureka-video
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --time=1:00:00
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com

# Setup conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eureka

# ---- Edit these ----
CHECKPOINT_PATH="checkpoints/EurekaPenSpinning.pth"
TASK_NAME="ShadowHand"
NUM_ENVS=1
VIDEO_LEN=600
# --------------------

cd isaacgymenvs/isaacgymenvs
python train.py \
  test=True \
  headless=False \
  force_render=True \
  capture_video=True \
  capture_video_len=${VIDEO_LEN} \
  capture_video_freq=1000000 \
  num_envs=${NUM_ENVS} \
  task=${TASK_NAME} \
  checkpoint=${CHECKPOINT_PATH}
