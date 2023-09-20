#!/usr/bin/env bash

# Set paths
DATASET_PATH="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_val"
EXPERIMENT_PATH="scratch/sc_experiments/one_percent/sc_100ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/one_percent/sc_75ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/one_percent/sc_50ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/one_percent/sc_25ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/one_percent/sc_best_eval"

PRETRAINED_PATH="scratch/sc_experiments/one_percent/sc_100ep_train/model_100.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/one_percent/sc_100ep_train/model_75.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/one_percent/sc_100ep_train/model_50.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/one_percent/sc_100ep_train/model_25.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/one_percent/sc_100ep_train/model_best.pth.tar"

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_PATH

# Run evaluation script
python -u ./src/cls_eval_one_percent.py \
-j 8 \
-b 16 \
--print-freq 16 \
--cls-size 1000 2000 4000 8000 \
--num-cls 4 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--tau 0.1 \
--use-bn \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}