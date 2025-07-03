import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import cv2
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_ssim import SSIM
import torch.fft as fft
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn_extra.cluster import KMedoids
from scipy.linalg import det
from sklearn.metrics.pairwise import rbf_kernel
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
    adjust_indices = torch.arange(0, T, 4, device=frames.device)
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



def color_histogram(img):
    """计算颜色直方图（HWC格式的uint8图像）"""
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def drop_video_frames(video_tensor, target_keep_ratio=0.75):
    """
    输入: 
        video_tensor - 形状为 (num_frames, C, H, W) 的PyTorch张量，值范围0-255，类型uint8
        target_keep_ratio - 保留帧的比例（默认0.75）
    输出: 
        保留后的视频张量
    """
    # 转换为numpy数组并调整维度为HWC
    frames_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    num_frames = frames_np.shape[0]
    target_keep = int(num_frames * target_keep_ratio)
    
    # 计算所有帧的直方图并过滤低信息帧
    valid_indices = []
    histograms = []
    for i, frame in enumerate(frames_np):
        hist = color_histogram(frame)
        if np.sum(hist > 0) > 10:  # 过滤纯色帧
            valid_indices.append(i)
            histograms.append(hist)
    
    # 动态调整相似性阈值以实现保留比例
    low, high = 0.7, 1.0
    best_threshold = 0.9
    best_kept = []
    
    # 二分法寻找最优阈值（最多迭代10次）
    for _ in range(10):
        mid = (low + high) / 2
        kept = []
        remaining = list(range(len(valid_indices)))
        
        while remaining and len(kept) < target_keep:
            current = remaining.pop(0)
            kept.append(current)
            # 移除相似帧
            to_remove = []
            for idx in remaining:
                # 计算相似性
                sim = np.dot(histograms[current], histograms[idx]) / (
                    np.linalg.norm(histograms[current]) * np.linalg.norm(histograms[idx])
                )
                if sim > mid:
                    to_remove.append(idx)
            remaining = [x for x in remaining if x not in to_remove]
        
        # 转换为原始索引
        current_kept = [valid_indices[i] for i in kept]
        # 补充不足的帧
        if len(current_kept) < target_keep:
            current_kept += [i for i in range(num_frames) if i not in current_kept][:target_keep-len(current_kept)]
        
        if len(current_kept) >= target_keep:
            best_threshold = mid
            best_kept = current_kept[:target_keep]
            low = mid
        else:
            high = mid
    
    # 按原始顺序排序并提取张量
    best_kept = sorted(list(set(best_kept)))  # 去重并排序
    return video_tensor[best_kept]



def select_protected_frames(video_tensor, threshold=0.47):
    """
    在ViT编码前对原始视频帧剪枝，加入相邻帧保护机制
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    规则：
    1. 计算相邻两帧的L1距离，将差异超过阈值的两帧都标记为保护帧
    2. 每四帧为一组，默认保留前三帧
    3. 每组第四帧如果是保护帧则保留，否则丢弃
    """
    num_frames = video_tensor.size(0)
    protected = set()

    # 计算相邻帧差异并标记保护帧
    for i in range(num_frames - 1):
        diff = torch.abs(video_tensor[i] - video_tensor[i+1]).mean()
        if diff > threshold:
            protected.add(i)
            protected.add(i+1)

    keep_indices = []
    
    # 遍历处理每个四帧组
    for start in range(0, num_frames, 4):
        # 添加前三帧
        group_end = min(start + 3, num_frames)
        keep_indices.extend(range(start, group_end))
        
        # 检查第四帧是否要保留
        fourth = start + 3
        if fourth < num_frames and fourth in protected:
            keep_indices.append(fourth)

    return video_tensor[keep_indices]


def select_protected_frames_threshold4(video_tensor, threshold=0.4):
    """
    在ViT编码前对原始视频帧剪枝，加入相邻帧保护机制
    输入形状: (num_frames, channel, width, height)
    输出形状: (pruned_frames, channel, width, height)
    规则：
    1. 计算相邻两帧的L1距离，将差异超过阈值的两帧都标记为保护帧
    2. 每四帧为一组，默认保留前三帧
    3. 每组第四帧如果是保护帧则保留，否则丢弃
    """
    num_frames = video_tensor.size(0)
    protected = set()

    # 计算相邻帧差异并标记保护帧
    for i in range(num_frames - 1):
        diff = torch.abs(video_tensor[i] - video_tensor[i+1]).mean()
        if diff > threshold:
            protected.add(i)
            protected.add(i+1)

    keep_indices = []
    
    # 遍历处理每个四帧组
    for start in range(0, num_frames, 4):
        # 添加前三帧
        group_end = min(start + 3, num_frames)
        keep_indices.extend(range(start, group_end))
        
        # 检查第四帧是否要保留
        fourth = start + 3
        if fourth < num_frames and fourth in protected:
            keep_indices.append(fourth)

    return video_tensor[keep_indices]


def dynamic_window_l1(video_tensor, threshold=0.4):
    frame_number = video_tensor.shape[0]
    if frame_number <= 1:
        return video_tensor

    # 计算相邻帧的L1差异（平均绝对误差）
    diffs = torch.mean(torch.abs(video_tensor[1:] - video_tensor[:-1]), dim=(1, 2, 3))

    # 找到所有连续四帧差异小的区域
    qualified = []
    for i in range(len(diffs) - 3):
        if torch.all(diffs[i:i+4] < threshold):
            qualified.append(i)

    # 收集所有静态帧
    static_frames = set()
    for i in qualified:
        for j in range(i, i + 4 + 1):  # 对应原始帧索引i到i+4
            if j < frame_number:
                static_frames.add(j)

    # 划分连续区段
    segments = []
    current_segment = []
    current_is_static = None
    for i in range(frame_number):
        is_static = i in static_frames
        if not current_segment:
            current_segment.append(i)
            current_is_static = is_static
        else:
            if is_static == current_is_static:
                current_segment.append(i)
            else:
                segments.append((current_is_static, current_segment))
                current_segment = [i]
                current_is_static = is_static
    if current_segment:
        segments.append((current_is_static, current_segment))

    # 生成保留的帧索引
    kept_indices = []
    for is_static, seg in segments:
        if is_static:
            # 静态区段：每隔一帧保留
            kept_indices.extend(seg[::2])
        else:
            # 动态区段：每隔四帧丢弃一帧
            kept_indices.extend([seg[i] for i in range(len(seg)) if (i % 5) != 4])


    print("kept_indices",kept_indices)        

    return video_tensor[kept_indices]


def select_frame_global_local(video_tensor, alpha=1.0, beta=1.0):
    """
    输入形状: (frame_number, channel, height, width)
    输出形状: (keep_num, channel, height, width)
    """
    F, C, H, W = video_tensor.shape
    
    # 计算全局L1（每帧像素绝对值之和）
    global_l1 = torch.sum(torch.abs(video_tensor), dim=(1, 2, 3))
    
    # 计算局部L1（与前一帧的绝对差之和）
    if F > 1:
        diff = torch.abs(video_tensor[1:] - video_tensor[:-1])
        local_l1 = torch.sum(diff, dim=(1, 2, 3))
        local_l1 = torch.cat([torch.zeros(1, device=video_tensor.device), local_l1])
    else:
        local_l1 = torch.zeros_like(global_l1)
    
    # 归一化处理
    global_norm = (global_l1 - global_l1.min()) / (global_l1.max() - global_l1.min() + 1e-12)
    local_norm = (local_l1 - local_l1.min()) / (local_l1.max() - local_l1.min() + 1e-12)
    
    # 结合得分
    combined_score = alpha * global_norm + beta * local_norm
    
    # 计算保留帧数并选择
    keep_num = max(1, int(0.75 * F))  # 至少保留1帧
    _, top_indices = torch.topk(combined_score, keep_num)
    top_indices, _ = torch.sort(top_indices)  # 保持原始顺序
    
    return video_tensor[top_indices]



def compute_adjacent_l1(video_tensor):
    """
    计算相邻两帧的L1分数
    :param video_tensor: 输入视频Tensor，形状为 (T, C, H, W)
    :return: L1分数Tensor，形状为 (T, H, W)
    """
    # 计算相邻帧的绝对差（结果形状：T-1, C, H, W）
    diff = torch.abs(video_tensor[1:] - video_tensor[:-1])

    # 对通道维度取L1（求和），形状变为 (T-1, H, W)
    l1 = diff.sum(dim=1)  # 如果用均值，改为 diff.mean(dim=1)

    # 第一帧补零填充，保持形状对齐
    zeros = torch.zeros((1, *l1.shape[1:]), 
                        dtype=l1.dtype, 
                        device=l1.device)
    return torch.cat([zeros, l1], dim=0)



def compute_l1_scores(video_tensor):
    """
    计算每一帧与前后相邻帧的L1距离
    Args:
        video_tensor: 输入视频张量，形状为 (frame_num, channels, height, width)
    
    Returns:
        l1_scores: 输出L1分数张量，形状为 (frame_num, height, width)
    """
    # 前向差异（当前帧与下一帧的差异）
    forward_diff = torch.zeros_like(video_tensor)
    forward_diff[:-1] = torch.abs(video_tensor[:-1] - video_tensor[1:])
    
    # 后向差异（当前帧与上一帧的差异）
    backward_diff = torch.zeros_like(video_tensor)
    backward_diff[1:] = torch.abs(video_tensor[1:] - video_tensor[:-1])
    
    # 合并差异并沿通道维度求和
    total_diff = forward_diff + backward_diff
    l1_scores = total_diff.sum(dim=1)  # 形状变为 (frame_num, height, width)
    
    return l1_scores


def compute_global_l1_scores(video_tensor):
    """
    计算每帧与其他所有帧的L1距离之和
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        L1分数张量，形状为 (T, H, W)，每个位置的值表示该帧在此位置
        与所有其他帧对应位置的绝对差之和
    """
    # 计算两两帧之间的绝对差
    diff = torch.abs(video_tensor.unsqueeze(1) - video_tensor.unsqueeze(0))  # (T, T, C, H, W)
    
    # 沿通道维度求和
    channel_sum = diff.sum(dim=2)  # (T, T, H, W)
    # channel_sum=diff[:, :, 2] 
    
    # 排除自身比较（对角线元素），求和得到最终结果
    l1_scores = channel_sum.sum(dim=1)  # (T, H, W)
    
    return l1_scores


def compute_global_l1_scores_channel_0(video_tensor):
    """
    计算每帧与其他所有帧的L1距离之和
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        L1分数张量，形状为 (T, H, W)，每个位置的值表示该帧在此位置
        与所有其他帧对应位置的绝对差之和
    """
    # 计算两两帧之间的绝对差
    diff = torch.abs(video_tensor.unsqueeze(1) - video_tensor.unsqueeze(0))  # (T, T, C, H, W)
    
    # 沿通道维度求和
    # channel_sum = diff.sum(dim=2)  # (T, T, H, W)
    channel_sum=diff[:, :, 0] 
    
    # 排除自身比较（对角线元素），求和得到最终结果
    l1_scores = channel_sum.sum(dim=1)  # (T, H, W)
    
    return l1_scores


def select_frames_based_on_50percent_scores_sum(video_tensor):
    """
    根据每帧的前50%区域得分筛选帧，丢弃得分最低的25%的帧
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        筛选后的视频张量
    """
    # 计算全局L1分数
    l1_scores = compute_global_l1_scores(video_tensor)  # (T, H, W)
    T, H, W = l1_scores.shape
    
    # 展平每个帧的H和W维度
    flat_scores = l1_scores.view(T, -1)  # (T, H*W)
    
    # 确定每个帧取前50%的元素数量（向上取整）
    k = (flat_scores.size(1) + 1) // 2
    
    # 计算每个帧的topk值之和作为得分
    topk_values, _ = torch.topk(flat_scores, k=k, dim=1, largest=True)
    frame_scores = topk_values.sum(dim=1)  # (T,)
    
    # 确定要删除的帧数
    num_remove = int(T * 0.25)
    if num_remove <= 0:
        return video_tensor  # 无需删除
    
    # 找到得分最低的num_remove个帧的索引
    _, remove_indices = torch.topk(frame_scores, k=num_remove, largest=False)
    
    # 创建保留掩码
    mask = torch.ones(T, dtype=torch.bool)
    mask[remove_indices] = False
    
    # 筛选保留的帧
    selected_video = video_tensor[mask]
    
    return selected_video


def select_frames_based_on_30percent_scores_sum(video_tensor):
    """
    根据每帧的前50%区域得分筛选帧，丢弃得分最低的25%的帧
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        筛选后的视频张量
    """
    # 计算全局L1分数
    l1_scores = compute_global_l1_scores(video_tensor)  # (T, H, W)
    T, H, W = l1_scores.shape
    
    # 展平每个帧的H和W维度
    flat_scores = l1_scores.view(T, -1)  # (T, H*W)
    
    # 确定每个帧取前50%的元素数量（向上取整）
    k = (flat_scores.size(1) + 1) // 3
    
    # 计算每个帧的topk值之和作为得分
    topk_values, _ = torch.topk(flat_scores, k=k, dim=1, largest=True)
    frame_scores = topk_values.sum(dim=1)  # (T,)
    
    # 确定要删除的帧数
    num_remove = int(T * 0.25)
    if num_remove <= 0:
        return video_tensor  # 无需删除
    
    # 找到得分最低的num_remove个帧的索引
    _, remove_indices = torch.topk(frame_scores, k=num_remove, largest=False)
    
    # 创建保留掩码
    mask = torch.ones(T, dtype=torch.bool)
    mask[remove_indices] = False
    
    # 筛选保留的帧
    selected_video = video_tensor[mask]
    
    return selected_video


def select_frames_based_on_50percent_scores_channel(video_tensor):
    """
    根据每帧的前50%区域得分筛选帧，丢弃得分最低的25%的帧
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        筛选后的视频张量
    """
    # 计算全局L1分数
    l1_scores = compute_global_l1_scores_channel_0(video_tensor)  # (T, H, W)
    T, H, W = l1_scores.shape
    
    # 展平每个帧的H和W维度
    flat_scores = l1_scores.view(T, -1)  # (T, H*W)
    
    # 确定每个帧取前50%的元素数量（向上取整）
    k = (flat_scores.size(1) + 1) // 2
    
    # 计算每个帧的topk值之和作为得分
    topk_values, _ = torch.topk(flat_scores, k=k, dim=1, largest=True)
    frame_scores = topk_values.sum(dim=1)  # (T,)
    
    # 确定要删除的帧数
    num_remove = int(T * 0.25)
    if num_remove <= 0:
        return video_tensor  # 无需删除
    
    # 找到得分最低的num_remove个帧的索引
    _, remove_indices = torch.topk(frame_scores, k=num_remove, largest=False)
    
    # 创建保留掩码
    mask = torch.ones(T, dtype=torch.bool)
    mask[remove_indices] = False
    
    # 筛选保留的帧
    selected_video = video_tensor[mask]
    
    return selected_video


def select_frames_based_on_30percent_scores_chanenel(video_tensor):
    """
    根据每帧的前50%区域得分筛选帧，丢弃得分最低的25%的帧
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        筛选后的视频张量
    """
    # 计算全局L1分数
    l1_scores = compute_global_l1_scores_channel_0(video_tensor)  # (T, H, W)
    T, H, W = l1_scores.shape
    
    # 展平每个帧的H和W维度
    flat_scores = l1_scores.view(T, -1)  # (T, H*W)
    
    # 确定每个帧取前50%的元素数量（向上取整）
    k = (flat_scores.size(1) + 1) // 3
    
    # 计算每个帧的topk值之和作为得分
    topk_values, _ = torch.topk(flat_scores, k=k, dim=1, largest=True)
    frame_scores = topk_values.sum(dim=1)  # (T,)
    
    # 确定要删除的帧数
    num_remove = int(T * 0.25)
    if num_remove <= 0:
        return video_tensor  # 无需删除
    
    # 找到得分最低的num_remove个帧的索引
    _, remove_indices = torch.topk(frame_scores, k=num_remove, largest=False)
    
    # 创建保留掩码
    mask = torch.ones(T, dtype=torch.bool)
    mask[remove_indices] = False
    
    # 筛选保留的帧
    selected_video = video_tensor[mask]
    
    return selected_video



def compute_global_l1_scores_select(video_tensor):
    """
    计算每帧与其他所有帧的L1距离的最大值
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        L1分数张量，形状为 (T, H, W)，每个位置的值表示该帧在此位置
        与所有其他帧对应位置的绝对差的最大值
    """
    # 计算两两帧之间的绝对差
    diff = torch.abs(video_tensor.unsqueeze(1) - video_tensor.unsqueeze(0))  # (T, T, C, H, W)
    
    # 沿通道维度求和（或根据需求改为其他操作）
    channel_sum = diff.sum(dim=2)  # (T, T, H, W)
    
    # 沿帧维度取最大值（替代原来的求和）
    l1_scores = channel_sum.max(dim=1)[0]  # (T, H, W)
    
    return l1_scores


def compute_global_l1_scores_max_pool(video_tensor):
    """
    计算每帧与其他所有帧的L1距离之和，并进行窗口最大池化
    Args:
        video_tensor: 输入视频张量，形状为 (T, C, H, W)
    
    Returns:
        L1分数张量，形状为 (T, H//2, W//2)，每个窗口的值为该窗口内的最大L1分数
    """
    # 计算两两帧之间的绝对差
    diff = torch.abs(video_tensor.unsqueeze(1) - video_tensor.unsqueeze(0))  # (T, T, C, H, W)
    
    # 沿通道维度求和
    channel_sum = diff.sum(dim=2)  # (T, T, H, W)
    
    # 排除自身比较（对角线元素），求和得到最终结果
    l1_scores = channel_sum.sum(dim=1)  # (T, H, W)
    
    # 添加通道维度并应用最大池化
    l1_scores = F.max_pool2d(l1_scores.unsqueeze(1),  # (T, 1, H, W)
                            kernel_size=2, 
                            stride=2).squeeze(1)      # (T, H//2, W//2)
    
    return l1_scores