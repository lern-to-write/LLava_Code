#!/bin/bash

# 记录清理前的内存使用情况
echo "清理前的内存使用情况："
free -h

# 同步文件系统缓冲区
sync

# 清理缓存
echo 3 > /proc/sys/vm/drop_caches

# 清理交换空间
swapoff -a && swapon -a

# 记录清理后的内存使用情况
echo "清理后的内存使用情况："
free -h