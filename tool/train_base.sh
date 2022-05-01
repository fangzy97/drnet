#!/bin/sh
# coding: utf-8
PARTITION=Segmentation

dataset=$1
model_name=$2
exp_name=$3
gpu=$4
exp_dir=exp/${model_name}/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=data/config/${dataset}/${dataset}_${model_name}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/train_base.sh tool/train_base.py model/${model_name}.py ${config} ${exp_dir}

CUDA_VISIBLE_DEVICES=${gpu}  python -u -m tool.train_base --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
