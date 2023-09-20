#!/usr/bin/env bash

DATASET_PATH="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_val"
EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/sc_800ep_cls_eval_superclass_lvl_5"
PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/sc_800ep_train/self-classifier.pth.tar"
mkdir -p $EXPERIMENT_PATH

python -u ./src/cls_eval.py \
--superclass 5 \
-j 8 \
-b 512 \
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