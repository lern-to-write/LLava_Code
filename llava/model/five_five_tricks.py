import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def compute_frame_mean_score_matrix_multi_gaussian_small_var_cahnnel(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=1, keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    return k_xy  # 直接返回核矩阵


    
def compute_video_mean_score_matrix_multi_gaussian_small_var_channel(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=(0,1), keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    return k_xy  # 直接返回核矩阵





def compute_frame_mean_score_cosine_multi_gaussian_small_var(input_tensor, token_per_frame=169, alphas=None, c=1.0, d=2):
    """使用多高斯核映射到高维空间计算余弦相似度"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=1, keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    # 计算自相似度K(x,x)和K(y,y)
    k_xx = sum([torch.exp(torch.zeros_like(l2_distance_square)) for _ in alphas])  # K(x,x)=Σexp(0)=len(alphas)
    k_yy = k_xx  # 因为avg_token是归一化的，||y-y||=0
    
    # 核空间中的余弦相似度
    token_scores = k_xy / torch.sqrt(k_xx * k_yy)
    
    return token_scores

def compute_video_mean_score_cosine_multi_gaussian_small_var(input_tensor, token_per_frame=169, alphas=None, c=1.0, d=2):
    """使用多高斯核映射到高维空间计算余弦相似度"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=(0,1), keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    # 计算自相似度K(x,x)和K(y,y)
    k_xx = sum([torch.exp(torch.zeros_like(l2_distance_square)) for _ in alphas])  # K(x,x)=Σexp(0)=len(alphas)
    k_yy = k_xx  # 因为avg_token是归一化的，||y-y||=0
    
    # 核空间中的余弦相似度
    token_scores = k_xy / torch.sqrt(k_xx * k_yy)
    
    return token_scores





def compute_frame_mean_score_cosine_by_kernal_small_var(input_tensor, token_per_frame=169, kernel_type='rbf', gamma=0.5, c=1.0, d=2):
    """使用核函数映射到高维空间计算余弦相似度"""
# 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim= 1, keepdim=True)  # (1, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 根据核类型计算相似度
    if kernel_type == 'rbf':
        # RBF核：K(x,y) = exp(-gamma * ||x-y||^2)
        squared_diff = torch.sum((frames - expanded_avg) ** 2, dim=2)
        k_xy = torch.exp(-gamma * squared_diff)
        k_xx = torch.ones_like(k_xy)  # RBF核的K(x,x)=1
        k_yy = torch.ones_like(k_xy)
        
    elif kernel_type == 'polynomial':
        # 多项式核：K(x,y) = (x·y + c)^d
        dot_product = torch.sum(frames * expanded_avg, dim=2)
        k_xy = (dot_product + c) ** d
        x_norm = torch.sum(frames ** 2, dim=2)
        k_xx = (x_norm + c) ** d
        y_norm = torch.sum(avg_token ** 2, dim=2)
        k_yy = (y_norm + c) ** d
    
    # 计算核空间中的余弦相似度
    token_scores = k_xy / torch.sqrt(k_xx * k_yy)
    
    return token_scores

def compute_video_mean_score_cosine_by_kernal_small_var(
    input_tensor,
    token_per_frame=169,
    gamma=0.5
):
    # 数据预处理
# 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // token_per_frame
    frames = selected_tensor.view(num_frame, token_per_frame, -1)  # [num_frame, 196, k]

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 单位球面归一化

    # 计算全局平均token
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)
    avg_expanded = avg_token.expand_as(frames)  # (T, N, D)

    # 优化后的RBF核计算
    def rbf_kernel(x, y, gamma):
        # 直接利用归一化后的性质: ||x-y||^2 = 2 - 2<x,y>
        pairwise_dot = torch.einsum('tnd,tnd->tn', x, y)  # 形状 (T, N)
        squared_dist = 2 - 2 * pairwise_dot  # 形状 (T, N)
        return torch.exp(-gamma * squared_dist)  # 直接输出目标形状

    # 计算核值矩阵
    kernel_matrix = rbf_kernel(frames, avg_expanded, gamma)
    return kernel_matrix  # 形状 (T, N)