#!/bin/bash

# 定义要切换的处理器列表
PROCESSORS=(   "select_frames_whith_posi_l1_scale_1" "select_frames_whith_posi_l1_scale_3" ) 

# 基础命令模板（保持你的原始参数不变）
BASE_CMD="accelerate launch --num_processes=2 -m lmms_eval --model llava_onevision --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen --tasks videomme --batch_size 1 --log_samples --log_samples_suffix llava_onevision"

# 遍历所有处理器
for processor in "${PROCESSORS[@]}"; do
    echo "▶▶ 开始执行处理器: $processor"
    
    # 动态生成带处理器后缀的输出路径
    OUTPUT_PATH="./logs/${processor}_results"
    
    # 执行命令（关键修改点）
    PROCESSOR=$processor $BASE_CMD \
        --output_path $OUTPUT_PATH
    
    echo "◀◀ 处理器 $processor 执行完成，日志保存在: $OUTPUT_PATH"
    echo "======================================================"
done