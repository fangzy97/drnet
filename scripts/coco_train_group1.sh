#!/usr/bin/env bash

cd ..

GPU_ID=5
GPOUP_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python train_frame_coco.py \
    --group=${GPOUP_ID} \
    --num_folds=4 \
    --arch=drnet \
    --batch_size=4 \
    --dataset=COCO \
    --lr=1e-5
