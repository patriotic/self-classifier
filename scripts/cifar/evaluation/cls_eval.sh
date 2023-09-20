#!/usr/bin/env bash

DATASET_PATH="scratch/cifar10"
#EXPERIMENT_PATH="scratch/sc_experiments/cifar/sc_70ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/cifar/sc_80ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/cifar/sc_90ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/cifar/sc_best_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/eval/sc_900_eval/sc_100ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/eval/sc_900_eval/sc_90ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/eval/sc_900_eval/sc_80ep_eval"
#EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/eval/sc_900_eval/sc_70ep_eval"
EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/eval/sc_900_eval/sc_60ep_eval"

#PRETRAINED_PATH="scratch/sc_experiments/cifar/sc_100ep_train/model_70.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/cifar/sc_100ep_train/model_80.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/cifar/sc_100ep_train/model_90.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/cifar/sc_100ep_train/model_best.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_900ep_train/model_100.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_900ep_train/model_90.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_900ep_train/model_80.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_900ep_train/model_70.pth.tar"
PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_900ep_train/model_60.pth.tar"

# Create experiment directory if it doesn't exist
mkdir -p $EXPERIMENT_PATH

# Run evaluation script
python -u ./src/cls_eval_cifar.py \
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