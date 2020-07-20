#!/usr/local/env bash
TF_CUDNN_USE_AUTOTUNE=0 python3 -u eval_scale.py \
--test_data_path=./debug_img/ \
--text_scale=512 \
--gpu_list=$1 \
--checkpoint_path=./checkpoints/resnet_v1_50-model.ckpt \
--output_dir=./results/ \
--resize_ratio=1.5 \
--no_write_images=False \
--max_side_len=2400 \
--network=resnet_v1_50
