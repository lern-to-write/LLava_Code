import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def compute_frame_mean_score_multi_gaussian(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
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
def compute_video_mean_score_multi_gaussian(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
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

def compute_video_mean_score_by_kernal(input_tensor, token_per_frame=169, gamma=0.5):
    """使用RBF核函数计算每个token与全局平均token的核矩阵"""
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token（跨所有帧和token）
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # (1, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算RBF核矩阵
    squared_diff = torch.sum((frames - expanded_avg) ** 2, dim=2)
    k_xy = torch.exp(-gamma * squared_diff)  # 直接返回核矩阵
    
    return k_xy

def compute_frame_mean_score_by_kernal(input_tensor, token_per_frame=169, gamma=0.5):
    """使用RBF核函数计算每个token与全局平均token的核矩阵"""
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token（跨所有帧和token）
    avg_token = frames.mean(dim=1, keepdim=True)  # (1, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算RBF核矩阵
    squared_diff = torch.sum((frames - expanded_avg) ** 2, dim=2)
    k_xy = torch.exp(-gamma * squared_diff)  # 直接返回核矩阵
    
    return k_xy



def compute_video_mean_score_by_kernal_v2(
    input_tensor,
    token_per_frame=169,
    gamma=0.5
):
    # 数据预处理
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (T, N, D)
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



def compute_frame_mean_score_by_kernal_v2(
    input_tensor,
    token_per_frame=169,
    gamma=0.5
):
    # 数据预处理
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (T, N, D)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 单位球面归一化

    # 计算全局平均token
    avg_token = frames.mean(dim= 1, keepdim=True)  # (1, 1, D)
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