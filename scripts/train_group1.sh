#!/usr/bin/env bash

cd ..

GPU_ID=1
GPOUP_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python train_frame.py \
    --group=${GPOUP_ID} \
    --num_folds=4 \
    --arch=drnet \
    --lr=1e-5

