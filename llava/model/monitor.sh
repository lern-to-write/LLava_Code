
#!/bin/bash
 
# GPU 检测时间间隔（秒）
CHECK_INTERVAL=30
 
# 显存使用阈值（MB）
MEMORY_THRESHOLD=500
 
# GPU 利用率阈值（百分比）
UTILIZATION_THRESHOLD=10
 
# 占卡程序路径 可以修改为要运行的python程序指令等
OCCUPY_SCRIPT="/obs/users/yiyu/new_env2/llava/model/test.sh"
 
# 检测显卡状态
check_gpu_status() {
    # 获取显存使用情况（MiB）
    memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
    
    # 获取 GPU 利用率（百分比）
    utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum += $1} END {print sum / NR}')
    
    # 输出当前状态
    echo "Current GPU memory usage: $1.1MB MiB"
    echo "Current GPU utilization: ${utilization}%"
 
    # 判断是否满足空卡条件，可以根据对gpu utils和memory的要求来单独设置限制
    if [[ $memory_usage -lt $MEMORY_THRESHOLD && ${utilization%%.*} -lt $UTILIZATION_THRESHOLD ]]; then
        return 0  # 空卡
    else
        return 1  # 非空卡
    fi
}
 
# 循环检测
while true; do
    echo "Checking GPU status..."
    
    if check_gpu_status; then
        echo "GPU is idle. Running occupy script..."
        bash "$OCCUPY_SCRIPT"
        # 也可以直接调用python文件，以其他指令方式实现命令
        break
    else
        echo "GPU is not idle. Checking again in ${CHECK_INTERVAL} seconds..."
    fi
    
    sleep $CHECK_INTERVAL
done
