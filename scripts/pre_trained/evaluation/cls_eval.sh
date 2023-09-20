#!/usr/bin/env bash

DATASET_PATH="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_val"
EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/sc_800ep_eval"
PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/sc_800ep_train/self-classifier.pth.tar"

mkdir -p $EXPERIMENT_PATH

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