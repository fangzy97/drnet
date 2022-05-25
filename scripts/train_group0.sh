#!/usr/bin/env bash

cd ..

GPU_ID=0
GPOUP_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python train_frame.py \
    --group=${GPOUP_ID} \
    --num_folds=4 \
    --arch=drnet \
    --lr=1e-5
