#!/bin/bash
export TASK_NAME=cola

python run_glue_alt.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 10.0 \
  --output_dir ./adapter/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer
