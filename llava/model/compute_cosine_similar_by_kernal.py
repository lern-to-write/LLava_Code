import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
def compute_frame_mean_score_multi_gaussian(input_tensor, token_per_frame=169, alphas=None, c=1.0, d=2):
    """使用多高斯核映射到高维空间计算余弦相似度"""
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
    
    # 计算自相似度K(x,x)和K(y,y)
    k_xx = sum([torch.exp(torch.zeros_like(l2_distance_square)) for _ in alphas])  # K(x,x)=Σexp(0)=len(alphas)
    k_yy = k_xx  # 因为avg_token是归一化的，||y-y||=0
    
    # 核空间中的余弦相似度
    token_scores = k_xy / torch.sqrt(k_xx * k_yy)
    from IPython import embed;embed()
    
    return token_scores

def compute_video_mean_score_multi_gaussian(input_tensor, token_per_frame=169, alphas=None, c=1.0, d=2):
    """使用多高斯核映射到高维空间计算余弦相似度"""
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
    
    # 计算自相似度K(x,x)和K(y,y)
    k_xx = sum([torch.exp(torch.zeros_like(l2_distance_square)) for _ in alphas])  # K(x,x)=Σexp(0)=len(alphas)
    k_yy = k_xx  # 因为avg_token是归一化的，||y-y||=0
    
    # 核空间中的余弦相似度
    token_scores = k_xy / torch.sqrt(k_xx * k_yy)
    
    return token_scores
def compute_frame_mean_score_by_kernal_v2(
    input_tensor,
    token_per_frame=169,
    kernel_type='rbf',
    **kwargs
):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (T, N, D)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    avg_token = frames.mean(dim= 1, keepdim=True)  # (1, 1, D)
    avg_expanded = avg_token.expand_as(frames)  # (T, N, D)


    def estimate_gamma(X):

        X_flat = X.view(-1, X.shape[-1])
        pairwise_distances = torch.cdist(X_flat, X_flat, p=2).pow(2)

        distances = pairwise_distances[pairwise_distances > 0]
        median_dist = torch.median(distances)
        return 1.0 / (2 * median_dist)

    gamma = estimate_gamma(frames)

    def linear_kernel(x, y):
        return torch.sum(x * y, dim=-1)

    def polynomial_kernel(x, y, c=1, d=2):
        return (torch.sum(x * y, dim=-1) + c) ** d

    def rbf_kernel(x, y, gamma):
        # 利用恒等式: ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
        kxx = torch.sum(x * x, dim=-1)
        kyy = torch.sum(y * y, dim=-1)
        kxy = torch.sum(x * y, dim=-1)
        return torch.exp(-gamma * (kxx + kyy - 2 * kxy))

    if kernel_type == 'linear':
        k_numerator = linear_kernel(frames, avg_expanded)
        k_x = linear_kernel(frames, frames)
        k_a = linear_kernel(avg_expanded, avg_expanded)
    elif kernel_type == 'polynomial':
        c = kwargs.get('c', 1)
        d = kwargs.get('d', 2)
        k_numerator = polynomial_kernel(frames, avg_expanded, c=c, d=d)
        k_x = polynomial_kernel(frames, frames, c=c, d=d)
        k_a = polynomial_kernel(avg_expanded, avg_expanded, c=c, d=d)
    elif kernel_type == 'rbf':
        gamma = kwargs.get('gamma', 1.0)
        k_numerator = rbf_kernel(frames, avg_expanded, gamma=gamma)
        k_x = rbf_kernel(frames, frames, gamma=gamma)
        k_a = rbf_kernel(avg_expanded, avg_expanded, gamma=gamma)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # 防止除零错误
    epsilon = 1e-12
    denominator = torch.sqrt(k_x * k_a).clamp(min=epsilon)

    # 计算核空间余弦相似度
    token_scores = k_numerator / denominator


    return token_scores
def compute_video_mean_score_by_kernal(input_tensor, token_per_frame=169, kernel_type='rbf', gamma=0.5, c=1.0, d=2):
    """使用核函数映射到高维空间计算余弦相似度"""
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # (1, 1, dim)
    
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


def compute_frame_mean_score_by_kernal(input_tensor, token_per_frame=169, kernel_type='rbf', gamma=0.5, c=1.0, d=2):
    """使用核函数映射到高维空间计算余弦相似度"""
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (num_frame, token_per_frame, dim)
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