#!/bin/bash

# 定义要切换的处理器列表
PROCESSORS=(
    "negtive_adaptive_ratio_frame_mean_half_small_channel"
    "positive_adaptive_ratio_frame_mean_half_small_channel"
    "positive_adaptive_ratio_frame_mean"
    "negtive_adaptive_ratio_frame_mean"
    "positive_adaptive_ratio_frame_mean_multi_gassion_matrix_half_cahnnel"
    "negtive_adaptive_ratio_frame_mean_multi_gassion_matrix_half_cahnnel"



)
# 基础命令模板（已移除环境变量设置）
BASE_CMD="accelerate launch --num_processes=5 --main_process_port 30004 -m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks videomme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid"


# 遍历所有处理器
for processor in "${PROCESSORS[@]}"; do
    echo "▶▶ 开始执行处理器: $processor"
    
    # 动态生成带处理器后缀的输出路径
    OUTPUT_PATH="./logs/${processor}_results"
    
    # 执行命令（关键修复点）
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 PROCESSOR=$processor $BASE_CMD \
        --output_path "$OUTPUT_PATH"
    
    echo "◀◀ 处理器 $processor 执行完成，日志保存在: $OUTPUT_PATH"
    echo "======================================================"
done