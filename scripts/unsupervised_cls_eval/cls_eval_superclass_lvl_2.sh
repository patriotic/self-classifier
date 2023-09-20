#!/usr/bin/env bash

DATASET_PATH="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_val"
EXPERIMENT_PATH="scratch/sc_experiments/sc_800ep_cls_eval_superclass_lvl_2"
PRETRAINED_PATH="scratch/sc_experiments/sc_800ep_train/model_800.pth.tar"
mkdir -p $EXPERIMENT_PATH

python -u ./src/cls_eval.py \
--superclass 2 \
-j 4 \
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