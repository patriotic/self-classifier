#!/usr/bin/env bash

EXPERIMENT_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_800ep_plus_train/"
PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_800ep_plus_train/self-classifier.pth.tar"
#PRETRAINED_PATH="scratch/sc_experiments/pre_trained_model/cifar/train/sc_800ep_plus_train/modified_model.pth.tar"

mkdir -p $EXPERIMENT_PATH

python -u ./src/modify_pre_trained_model.py \
--syncbn_process_group_size 22 \
-j 32 \
-b 372 \
--print-freq 16 \
--epochs 800 \
--lr 3.2 \
--start-warmup 0.2 \
--final-lr 0.0032 \
--lars \
--sgd \
--cos \
--wd 1e-6 \
--cls-size 1000 2000 4000 8000 \
--new-cls-size 10 20 40 80 \
--num-cls 4 \
--queue-len 262144 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--row-tau 0.1 \
--col-tau 0.05 \
--global-crops-scale 0.4 1.0 \
--local-crops-scale 0.05 0.4 \
--local-crops-number 6 \
--use-bn \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
""