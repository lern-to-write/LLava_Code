import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import cv2
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_ssim import SSIM
def select_frames_before_vit(input_tensor, retention_ratio=0.75):
    """
    在ViT之前减少视频帧数量
    输入形状: (num_frames, channel, height, width)
    输出形状: (k_frames, channel, height, width)
    """
    num_frames = input_tensor.shape[0]
    
    # 将每个帧展平为向量并计算通道平均值 (替代token特征)
    # 结果形状: (num_frames, channel)
    frame_features = torch.mean(input_tensor.view(num_frames, -1,3), dim=1)
    
    # 计算全局视频平均特征
    video_mean = torch.mean(frame_features, dim=0, keepdim=True)  # (1, channel)
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(
        frame_features,
        video_mean.expand_as(frame_features),
        dim=1
    )
    # 确定保留帧数（至少1帧）
    k_frames = max(1, int(round(retention_ratio * num_frames)))
    # 选择相似度最小的帧（与全局特征差异大的帧）
    _, indices = torch.topk(similarities, k=k_frames, largest=False)
    
    # 保持原始时间顺序
    sorted_indices, _ = torch.sort(indices)
    
    # 返回选择后的帧
    return input_tensor[sorted_indices]


def select_frame_by_keep_three_before_vit(video_tensor):
    """
    在ViT编码前对原始视频帧剪枝
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    规则：每四帧保留前三帧（从第0帧开始计数）
    """
    total_frames = video_tensor.size(0)
    keep_indices = []

    # 以4帧为步长遍历所有帧
    for start_idx in range(0, total_frames, 4):
        # 计算当前组的结束索引（不包含）
        end_idx = min(start_idx + 3, total_frames)
        # 添加当前组需要保留的索引
        keep_indices.extend(range(start_idx, end_idx))

    # 根据索引选择需要保留的帧
    pruned_video = video_tensor[keep_indices]
    return pruned_video



def select_frame_by_keep_two_before_vit(video_tensor):
    """
    在ViT编码前对原始视频帧剪枝
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    规则：每三帧保留前两帧（从第0帧开始计数）
    """
    total_frames = video_tensor.size(0)
    keep_indices = []

    # 以3帧为步长遍历所有帧
    for start_idx in range(0, total_frames, 3):
        # 计算当前组的结束索引（不包含）
        end_idx = min(start_idx + 2, total_frames)
        # 添加当前组需要保留的索引
        keep_indices.extend(range(start_idx, end_idx))

    # 根据索引选择需要保留的帧
    pruned_video = video_tensor[keep_indices]
    return pruned_video


def select_frame_by_keep_two_then_L1_before_vit(video_tensor):
    """
    在ViT编码前对原始视频帧剪枝
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    新规则：动态选择需要drop的帧，使得最终drop总量与原始策略相同
    """
    total_frames = video_tensor.size(0)
    
    # 计算原始策略应drop的帧数
    original_drop_num = total_frames // 4
    
    # 候选帧：所有可能的第三帧（索引2,5,8...）
    candidate_indices = list(range(2, total_frames, 3))
    
    # 计算候选帧与前一帧的L1距离
    distances = []
    for idx in candidate_indices:
        if idx >= 1:  # 确保有前一帧
            diff = torch.abs(video_tensor[idx] - video_tensor[idx-1]).sum()
            distances.append( (idx, diff.item()) )
    
    # 按距离升序排序，优先drop变化最小的帧
    sorted_candidates = sorted(distances, key=lambda x: x[1])
    
    # 确定实际需要drop的帧（最多不超过候选数量）
    drop_num = min(original_drop_num, len(sorted_candidates))
    drop_indices = {x[0] for x in sorted_candidates[:drop_num]}
    
    # 构建保留索引（排除要drop的帧）
    keep_indices = [i for i in range(total_frames) if i not in drop_indices]
    
    return video_tensor[keep_indices]




def select_frames_whith_neg_l1_scale(frames: torch.Tensor,score_scale=0.5) -> torch.Tensor:
    """
    视频帧压缩函数
    输入参数：
        frames : torch.Tensor 形状为 (num_frames, channel, width, height)
    返回：
        compressed_frames : torch.Tensor 压缩后的帧序列
    """
    T = frames.shape[0]
    if T == 0:
        return torch.empty_like(frames)
    
    # 步骤1：计算L1距离分数
    scores = torch.zeros(T, device=frames.device)
    if T > 1:
        # 计算相邻帧差异（自动广播到所有通道和空间维度）
        diff = frames[1:] - frames[:-1]
        # 计算L1距离：绝对值求和（跨通道、宽、高维度）
        l1_dist = torch.abs(diff).mean(dim=(1, 2, 3))  # 形状 (T-1,)
        scores[1:] = l1_dist  # 第0帧分数保持0
    
    # 步骤2：调整特定位置的分数
    # 生成需要调整的索引（3,7,11,...）
    adjust_indices = torch.arange(3, T, 4, device=frames.device)
    scores[adjust_indices] *= score_scale  # 分数减半
    
    # 步骤3：选择保留的帧
    # 计算需要保留的帧数（四分之三向上取整）
    k = (3 * T + 3) // 4  # 等价于 math.ceil(0.75*T)
    
    # 处理T=0的特殊情况（理论上不会出现）
    if k <= 0:
        return frames[0:0]  # 返回空tensor
    
    # 获取分数最高的k个索引
    _, topk_indices = torch.topk(-scores, k=k)  # 用负号实现升序排列
    topk_indices, _ = torch.sort(topk_indices)  # 按时间顺序排序
    
    # 步骤4：提取保留的帧
    return frames[topk_indices]
def select_frames_whith_posi_l1_scale_1(frames: torch.Tensor,score_scale=0.1) -> torch.Tensor:
    """
    视频帧压缩函数
    输入参数：
        frames : torch.Tensor 形状为 (num_frames, channel, width, height)
    返回：
        compressed_frames : torch.Tensor 压缩后的帧序列
    """
    T = frames.shape[0]
    if T == 0:
        return torch.empty_like(frames)
    
    # 步骤1：计算L1距离分数
    scores = torch.zeros(T, device=frames.device)
    if T > 1:
        # 计算相邻帧差异（自动广播到所有通道和空间维度）
        diff = frames[1:] - frames[:-1]
        # 计算L1距离：绝对值求和（跨通道、宽、高维度）
        l1_dist = torch.abs(diff).mean(dim=(1, 2, 3))  # 形状 (T-1,)
        scores[1:] = l1_dist  # 第0帧分数保持0
    
    # 步骤2：调整特定位置的分数
    # 生成需要调整的索引（3,7,11,...）
    adjust_indices = torch.arange(3, T, 4, device=frames.device)
    scores[adjust_indices] *= score_scale  # 分数减半
    
    # 步骤3：选择保留的帧
    # 计算需要保留的帧数（四分之三向上取整）
    k = (3 * T + 3) // 4  # 等价于 math.ceil(0.75*T)
    
    # 处理T=0的特殊情况（理论上不会出现）
    if k <= 0:
        return frames[0:0]  # 返回空tensor
    
    # 获取分数最高的k个索引
    _, topk_indices = torch.topk(scores, k=k)  
    topk_indices, _ = torch.sort(topk_indices)  # 按时间顺序排序
    
    # 步骤4：提取保留的帧
    return frames[topk_indices]
def select_frames_whith_posi_l1_scale_3(frames: torch.Tensor,score_scale=0.3) -> torch.Tensor:
    """
    视频帧压缩函数
    输入参数：
        frames : torch.Tensor 形状为 (num_frames, channel, width, height)
    返回：
        compressed_frames : torch.Tensor 压缩后的帧序列
    """
    T = frames.shape[0]
    if T == 0:
        return torch.empty_like(frames)
    
    # 步骤1：计算L1距离分数
    scores = torch.zeros(T, device=frames.device)
    if T > 1:
        # 计算相邻帧差异（自动广播到所有通道和空间维度）
        diff = frames[1:] - frames[:-1]
        # 计算L1距离：绝对值求和（跨通道、宽、高维度）
        l1_dist = torch.abs(diff).mean(dim=(1, 2, 3))  # 形状 (T-1,)
        scores[1:] = l1_dist  # 第0帧分数保持0
    
    # 步骤2：调整特定位置的分数
    # 生成需要调整的索引（3,7,11,...）
    adjust_indices = torch.arange(3, T, 4, device=frames.device)
    scores[adjust_indices] *= score_scale  # 分数减半
    
    # 步骤3：选择保留的帧
    # 计算需要保留的帧数（四分之三向上取整）
    k = (3 * T + 3) // 4  # 等价于 math.ceil(0.75*T)
    
    # 处理T=0的特殊情况（理论上不会出现）
    if k <= 0:
        return frames[0:0]  # 返回空tensor
    
    # 获取分数最高的k个索引
    _, topk_indices = torch.topk(scores, k=k)  
    topk_indices, _ = torch.sort(topk_indices)  # 按时间顺序排序
    
    # 步骤4：提取保留的帧
    return frames[topk_indices]
def select_frame_by_keep_three_before_vit(video_tensor):
    """
    严格保留总帧数的3/4（四舍五入）的改进版
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    """
    total_frames = video_tensor.size(0)
    
    # 计算需要保留的帧数（四舍五入）
    k = max(1, int(round(total_frames * 0.75)))  # 至少保留1帧
    
    keep_indices = []
    
    # 遍历每组4帧，动态调整最后一组的保留数量
    for start_idx in range(0, total_frames, 4):
        # 计算当前组最多可保留的帧数
        remaining = k - len(keep_indices)
        if remaining <= 0:
            break
        
        # 确定本组实际保留的帧数
        group_keep = min(3, remaining)  # 每组最多保留3帧
        
        # 计算本组实际结束位置
        end_idx = min(start_idx + group_keep, total_frames)
        
        # 添加本组保留的索引
        keep_indices.extend(range(start_idx, end_idx))

    return video_tensor[keep_indices]


def compress_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    视频帧压缩函数
    输入参数：
        frames : torch.Tensor 形状为 (num_frames, channel, width, height)
    返回：
        compressed_frames : torch.Tensor 压缩后的帧序列
    """
    T = frames.shape[0]
    if T == 0:
        return torch.empty_like(frames)
    
    # 步骤1：计算L1距离分数
    scores = torch.zeros(T, device=frames.device)
    if T > 1:
        # 计算相邻帧差异（自动广播到所有通道和空间维度）
        diff = frames[1:] - frames[:-1]
        # 计算L1距离：绝对值求和（跨通道、宽、高维度）
        l1_dist = torch.abs(diff).sum(dim=(1, 2, 3))  # 形状 (T-1,)
        scores[1:] = l1_dist  # 第0帧分数保持0

    # 步骤2：窗口化处理
    for window_start in range(0, T, 4):  # 步长4的滑动窗口
        window_end = min(window_start + 4, T)
        window_scores = scores[window_start:window_end]
        
        if len(window_scores) == 0:
            continue
        
        # 步骤3：找到窗口内最小分数并减半
        min_score = torch.min(window_scores)
        mask = (window_scores == min_score)
        # 直接修改原始scores张量
        scores[window_start:window_end][mask] *= 0.5

    # 步骤4：计算需要丢弃的帧数
    k = T // 4  # 丢弃总数的四分之一（向下取整）
    if k == 0:
        return frames.clone()  # 当帧数小于4时直接返回原始数据

    # 步骤5：选择要保留的帧
    # 找到分数最大的T-k帧（即丢弃分数最小的k帧）
    _, keep_indices = torch.topk(scores, k=T-k, largest=True)
    keep_indices, _ = torch.sort(keep_indices)  # 保持原始时序

    return frames[keep_indices]
