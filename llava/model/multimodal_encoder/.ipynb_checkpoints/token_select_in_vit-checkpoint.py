'''
Copyright (2024) Hefei University of Technology. 
Developers: liangtao shi, ting liu.

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
'''

import math
from typing import Callable, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np



def reduce_frames_by_small_l1(video_tensor):
    """
    Args:
        video_tensor: shape (frame_num, token_num, hidden_size)
        
    Returns:
        new_video_tensor: 保留3/4帧后的结果
    """
    # 计算相邻帧的L1距离
    diff = video_tensor[1:] - video_tensor[:-1]  # 相邻帧差值
    dists = torch.abs(diff).mean(dim=(1, 2))     # 计算L1距离（按token和hidden维度求和）
    
    # 计算需要丢弃的帧数（总帧数的1/4）
    total_frames = video_tensor.size(0)
    discard_num = total_frames // 2
    
    # 找到距离最小的discard_num个帧（这些帧将被丢弃）
    _, sorted_indices = torch.topk(dists, k=discard_num, largest=False)
    sorted_indices.sort()
    
    return video_tensor[sorted_indices]

def select_frames_by_l1(video_tensor):
    # 获取总帧数
    T = video_tensor.size(0)
    
    # 处理边界情况（帧数小于等于1时返回空张量）
    if T <= 1:
        return torch.empty(0, *video_tensor.shape[1:], device=video_tensor.device)
    
    # 计算相邻帧之间的绝对差
    diffs = video_tensor[1:] - video_tensor[:-1]
    
    # 计算每对相邻帧的L1距离（沿token和channel维度求和）
    l1_distances = torch.sum(torch.abs(diffs), dim=(1, 2))
    
    # 确定需要保留的帧数（总帧数的一半）
    k = T // 2
    
    # 处理不需要保留任何帧的情况
    if k <= 0:
        return torch.empty(0, *video_tensor.shape[1:], device=video_tensor.device)
    
    # 获取L1距离最大的k个帧的索引
    _, topk_indices = torch.topk(l1_distances, k)
    
    # 将索引转换为原始视频帧的索引，并保持时间顺序
    selected_indices = topk_indices + 1  # 转换为原始视频索引
    sorted_indices, _ = torch.sort(selected_indices)  # 保持时间顺序
    
    # 选择对应的帧
    selected_frames = video_tensor[sorted_indices-1]#[ 0,  4,  6,  7,  8,  9, 10, 11]
    
    return sorted_indices-1

def select_frames_index_by_l1(video_tensor, threshold=0.25):
    T = video_tensor.size(0)
    
    if T <= 1:
        return torch.empty(0, device=video_tensor.device, dtype=torch.long)
    
    diffs = video_tensor[1:] - video_tensor[:-1]
    l1_distances = torch.mean(torch.abs(diffs), dim=(1, 2))
    valid_indices = torch.where(l1_distances > threshold)[0]
    selected_indices = valid_indices + 1
    
    # 添加第0帧并保持排序
    zero_frame = torch.tensor([0], device=video_tensor.device, dtype=torch.long)
    selected_indices = torch.cat([zero_frame, selected_indices])
    selected_indices = torch.sort(selected_indices)[0]
    
    return selected_indices
def select_base_frames_index(image_feature, threshold=0.45):
    frame_number = image_feature.shape[0]
    base_indices = [0]
    current_base_index = 0  # 初始帧索引为0
    
    for i in range(frame_number - 1):
        # 当前帧和下一帧
        current_frame = image_feature[current_base_index]
        next_frame = image_feature[i+1]
        
        # 计算L1距离
        l1_distance = torch.abs(current_frame - next_frame)
        l1_score = l1_distance.mean().item()
        print("l1_score",l1_score)
        
        if l1_score > threshold:
            # 超过阈值时记录新base帧索引
            new_base_index = i + 1
            base_indices.append(new_base_index)
            current_base_index = new_base_index  # 更新当前基准帧
    
    return base_indices



def select_frames_whith_small_cosine_similarity_key_frame_mean(hidden_states,mean_head_k):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.25
    tokens_per_frame = mean_head_k.shape[1]
    num_frame = mean_head_k.shape[0] 
    frames = mean_head_k
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = similarities.sum(dim=1)  # 形状 (num_frames,)
    
    # 计算需要保留的帧数（四舍五入）
    k_frames = int(round((1 - prune_ratio) * total_frames))
    
    # 获取得分最高的k_frames个帧的索引
    _, top_indices = torch.topk(frame_scores, k=k_frames, largest=False)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(top_indices, dim=0)
    
    # 收集保留的帧（包含所有token）
    selected_frames = hidden_states[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames
    
    return pruned_tensor

def select_frames_whith_big_cosine_similarity_key_frame_mean(hidden_states,mean_head_k):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.25
    tokens_per_frame = mean_head_k.shape[1]
    num_frame = mean_head_k.shape[0] 
    frames = mean_head_k
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = similarities.sum(dim=1)  # 形状 (num_frames,)
    
    # 计算需要保留的帧数（四舍五入）
    k_frames = int(round((1 - prune_ratio) * total_frames))
    
    # 获取得分最高的k_frames个帧的索引
    _, top_indices = torch.topk(frame_scores, k=k_frames, largest=True)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(top_indices, dim=0)
    
    # 收集保留的帧（包含所有token）
    selected_frames = hidden_states[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames
    
    return pruned_tensor

def select_frames_whith_small_cosine_similarity_token_frame_mean(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.5
    tokens_per_frame = input_tensor.shape[1]
    num_frame = input_tensor.shape[0]
    frames = input_tensor
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = similarities.mean(dim=1)  # 形状 (num_frames,)
    
    # 计算需要保留的帧数（四舍五入）
    k_frames = int(round((1 - prune_ratio) * total_frames))
    
    # 获取得分最第的k_frames个帧的索引
    _, top_indices = torch.topk(frame_scores, k=k_frames, largest=False)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(top_indices, dim=0)
    
    # 收集保留的帧（包含所有token）
    selected_frames = frames[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames#[ 0,  6, 10, 11, 12, 13, 14, 15]
    return pruned_tensor

def select_frames_whith_big_cosine_similarity_token_frame_mean(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.5
    tokens_per_frame = input_tensor.shape[1]
    num_frame = input_tensor.shape[0]
    frames = input_tensor
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = similarities.mean(dim=1)  # 形状 (num_frames,)
    
    # 计算需要保留的帧数（四舍五入）
    k_frames = int(round((1 - prune_ratio) * total_frames))
    
    # 获取得分最高的k_frames个帧的索引
    _, top_indices = torch.topk(frame_scores, k=k_frames, largest=True)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(top_indices, dim=0)
    
    # 收集保留的帧（包含所有token）
    selected_frames = frames[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames

    
    return pruned_tensor



def filter_low_info_frames(video_tensor):
    """
    video_tensor: 输入Tensor，形状为 [frame_number, token_number, channel]
    返回值：过滤后的Tensor，丢弃信息量最低的25%帧，并保持原始顺序
    """
    frame_info = []
    F, T, C = video_tensor.shape
    
    # 计算每个帧的信息量
    for f in range(F):
        frame = video_tensor[f, :, :].float()  # [T, C]
        U, S, _ = torch.linalg.svd(frame, full_matrices=False)
        contributions = torch.abs(U) * S.unsqueeze(0)  # [T, r]
        total_info = contributions.sum().item()  # 帧总信息量
        frame_info.append(total_info)
    
    # 按信息量排序，保留前75%
    sorted_indices = sorted(range(F), key=lambda i: frame_info[i], reverse=True)
    num_keep = int(0.5 * F)
    kept_indices = sorted_indices[:num_keep]
    
    # 按原始顺序排序
    kept_indices.sort()
    
    # 提取保留的帧
    filtered_tensor = video_tensor[kept_indices, :, :]# [0, 1, 4, 6, 8, 9, 10, 11]
    
    return filtered_tensor



def process_video_frames(video_tensor, ratio=0.5):
    """
    Args:
        video_tensor: (frame_num, token_num, channel)
        ratio: 保留比例 (保留分数最小的ratio比例帧)
    Returns:
        selected_frames: 保留的帧组成的张量
    """
    frame_num, token_num, channel = video_tensor.shape
    
    # 检查token_num是否为平方数
    H = int(math.sqrt(token_num))
    if H * H != token_num:
        raise ValueError("Token number must be a perfect square for 2D transpose")
    W = H

    scores = []
    for i in range(1, frame_num):
        # 获取当前帧并转置
        current_frame = video_tensor[i]
        
        # 获取前一帧
        prev_frame = video_tensor[i-1]
        
        # 计算余弦相似度
        sim = F.cosine_similarity(current_frame, prev_frame, dim=1)
        scores.append(sim.mean().item())

    # 计算需要保留的帧数
    keep_num = max(1, int(ratio * (frame_num - 1)))
    
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])[:keep_num]

    # 按原始顺序排列索引
    sorted_indices.sort()

    # 如果需要保留第一帧，可以将其加入索引

    selected_indices = [0] + [i + 1 for i in sorted_indices]

    selected_frames = video_tensor[selected_indices]# [0, 7, 10, 11, 12, 13, 14, 15]

    
    return selected_frames




def resize_and_flatten_features(original_tensor, target_size=(14, 14)):

    # 验证输入形状
    assert original_tensor.shape[1] == 729, "输入特征维度应为729"
    
    # Step1: 重塑为2D结构
    reshaped = original_tensor.view(-1, 1, 27, 27)  # [batch, 1, 27, 27]
    
    # Step2: 双线性插值
    interpolated = F.interpolate(
        reshaped,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )  # [batch, 1, 14, 14]
    
    # Step3: 展平特征
    result = interpolated.view(original_tensor.size(0), -1)  # [batch, 196]
    
    return result



def resize_key_features(original_tensor, target_size=(14, 14)):

    assert original_tensor.shape[1] == 729, "输入特征的第二维度应为729"
    
    batch_size, _, dim = original_tensor.shape
    
    # Step1: 重塑为2D结构 [batch, 1, 27, 27]
    reshaped = original_tensor.permute(0, 2, 1).view(batch_size, dim, 27, 27)
    
    # Step2: 双线性插值 [batch, dim, 14, 14]
    interpolated = F.interpolate(
        reshaped,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    # Step3: 展平特征 [batch, dim, 196]
    flattened = interpolated.view(batch_size, dim, -1)
    
    # 调整维度顺序为 [batch, 196, dim]
    result = flattened.permute(0, 2, 1)
    
    return result   