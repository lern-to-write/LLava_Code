#!/bin/bash

# 第一个任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes=6 --main_process_port 30002 \
  -m lmms_eval \
  --model llava_vid \
  --model_args "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average" \
  --tasks perceptiontest_val_mc \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path ./logs/

# 等待第一个任务完成后自动执行第二个
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes=6 --main_process_port 30002 \
  -m lmms_eval \
  --model llava_vid \
  --model_args "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average" \
  --tasks nextqa_mc_test \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path ./logs/

# 等待第二个任务完成后自动执行第三个
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  accelerate launch --num_processes=6 --main_process_port 30002 \
  -m lmms_eval \
  --model llava_vid \
  --model_args "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average" \
  --tasks mvbench \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_vid \
  --output_path ./logs/