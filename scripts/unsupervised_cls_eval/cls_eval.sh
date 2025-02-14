#!/usr/bin/env bash

DATASET_PATH="scratch/cifar10"
EXPERIMENT_PATH="scratch/sc_experiments/sc_50ep_cifar_eval"
PRETRAINED_PATH="scratch/sc_experiments/sc_100ep_cifar_eval/model_50.pth.tar"

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_PATH

# Run evaluation script
python -u ./src/cls_eval.py \
-j 8 \
-b 512 \
--print-freq 16 \
--cls-size 10 20 40 80 \
--num-cls 4 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--tau 0.1 \
--use-bn \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}