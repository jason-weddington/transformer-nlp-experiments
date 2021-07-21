#!/bin/bash
export DATASET_NAME=glue
export DATASET_CONFIG_NAME=cola

python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir ./adapter/mlm/$DATASET_NAME \
    --overwrite_output_dir
    --train_adapter \
    --adapter_config "pfeiffer+inv"
