#!/usr/bin/env bash

DATASET_PATH="/content/drive/MyDrive/RL/datasets/one_percent"
EXPERIMENT_PATH="/content/drive/MyDrive/RL/scratch/sc_experiments/sc_100ep_train"
mkdir -p $EXPERIMENT_PATH

python -u ./src/train.py \
--syncbn_process_group_size 22 \
-j 32 \
-b 372 \
--print-freq 16 \
--epochs 100 \
--lr 4.8 \
--start-warmup 0.3 \
--final-lr 0.0048 \
--lars \
--sgd \
--cos \
--wd 1e-6 \
--cls-size 1000 2000 4000 8000 \
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
${DATASET_PATH}