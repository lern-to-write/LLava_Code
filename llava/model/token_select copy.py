import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



def generate_scales_whith_mean_video_score(input_tensor,base_scale=0.5,temperature=0.1):
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = -similarities.sum(dim=1)  # 形状 (num_frames,)
    frame_scores = frame_scores.cpu().numpy()
    # Normalize the scores and apply softmax
    shifted_scores = (frame_scores - np.max(frame_scores)) / temperature
    exp_scores = np.exp(shifted_scores)
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
    # Calculate scales ensuring no scale exceeds 1
    if np.sum(softmax_scores) == 0:
        scales = [base_scale] * num_frame
    else:
        scales = base_scale * (1 + softmax_scores - np.mean(softmax_scores))
        scales = np.clip(scales, None, 1.0)
    return  scales

def generate_scales_from_video_score(token_scores, base_scale=0.5, temperature=0.1):
    """根据token分数生成帧级缩放系数"""
    # 计算帧级得分（负相似度和）
    # frame_number = token_scores.shape[0] // 196
    # token_per_frame=196
    # token_scores=token_scores.view(frame_number,token_per_frame)
    frame_scores = token_scores.sum(dim=1).cpu().numpy()
    
    # 应用温度缩放和softmax
    shifted_scores = (frame_scores - np.max(frame_scores)) / temperature
    exp_scores = np.exp(shifted_scores)
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
    
    # 计算最终缩放系数
    if np.sum(softmax_scores) == 0:
        scales = [base_scale] * token_scores.shape[0]
    else:
        scales = base_scale * (1 + softmax_scores - np.mean(softmax_scores))
        scales = np.clip(scales, None, 1.0)
    return scales

def generate_scales_from_frame_video_score(frame_scores, base_scale=0.5, temperature=0.001):
    """根据token分数生成帧级缩放系数"""
    # 计算帧级得分（负相似度和）
    # frame_number = token_scores.shape[0] // 196
    # token_per_frame=196
    # token_scores=token_scores.view(frame_number,token_per_frame)
    
    # 应用温度缩放和softmax
    shifted_scores = (frame_scores - torch.max(frame_scores)) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)
    

    scales = base_scale * (1 + softmax_scores - torch.mean(softmax_scores))
    scales = torch.clip(scales, None, 1.0)
    return scales

def compute_video_mean_score(input_tensor,token_per_frame=196):
    """计算每个token的余弦相似度分数"""
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, dim)
    token_scores = F.cosine_similarity(frames, avg_token.expand_as(frames), dim=2)
    # video_mean_score=token_scores.flatten() 
    return token_scores


def compute_frame_mean_score_by_kernal_v2(
    input_tensor,
    token_per_frame=169,
    kernel_type='rbf',
    **kwargs
):
    """
    计算每个 token 在核空间中的余弦相似度分数
    
    参数:
        input_tensor (Tensor): 输入张量，形状为 (N, D)
        token_per_frame (int): 每帧包含的 token 数量
        kernel_type (str): 核函数类型 ('linear', 'polynomial', 'rbf')
        **kwargs: 核函数参数，如 gamma, c, d 等
    返回:
        Tensor: 每个 token 的核空间余弦相似度分数
    """
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (T, N, D)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    avg_token = frames.mean(dim= 1, keepdim=True)  # (1, 1, D)
    avg_expanded = avg_token.expand_as(frames)  # (T, N, D)


    def estimate_gamma(X):
        # X: shape (T, N, D) → flatten to (T*N, D)
        X_flat = X.view(-1, X.shape[-1])
        # 计算所有点两两之间的平方欧氏距离
        pairwise_distances = torch.cdist(X_flat, X_flat, p=2).pow(2)
        # 取非零距离的中位数
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
    kernel_type='rbf',
    **kwargs
):
    """
    计算每个 token 在核空间中的余弦相似度分数
    
    参数:
        input_tensor (Tensor): 输入张量，形状为 (N, D)
        token_per_frame (int): 每帧包含的 token 数量
        kernel_type (str): 核函数类型 ('linear', 'polynomial', 'rbf')
        **kwargs: 核函数参数，如 gamma, c, d 等
    返回:
        Tensor: 每个 token 的核空间余弦相似度分数
    """
    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)  # (T, N, D)
    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面

    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)
    avg_expanded = avg_token.expand_as(frames)  # (T, N, D)

    def linear_kernel(x, y):
        return torch.sum(x * y, dim=-1)

    def polynomial_kernel(x, y, c=1, d=2):
        return (torch.sum(x * y, dim=-1) + c) ** d

    def rbf_kernel(x, y, gamma=0.5):
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
def compute_video_mean_score_small_var(input_tensor,token_per_frame=169):
    """计算每个token在方差小前一半通道上的余弦相似度分数"""
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
    
    # 在筛选后的通道上计算均值
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # [1, 1, k]
    
    # 计算余弦相似度
    token_scores = F.cosine_similarity(frames, avg_token.expand_as(frames), dim=2)
    
    return token_scores


def compute_frame_mean_score_small_var(input_tensor,token_per_frame=169):
    """计算每个token在方差最小前一半通道上的余弦相似度分数"""
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
    
    # 在筛选后的通道上计算均值
    avg_token = frames.mean(dim=1, keepdim=True)  # [num_frame, 1, k]
    
    # 计算余弦相似度
    token_scores = F.cosine_similarity(frames, avg_token.expand_as(frames), dim=2)
    
    return token_scores




def compute_frame_mean_score(input_tensor):
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)

    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, 896)
    frame_mean_score = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    # frame_mean_score=similarities.flatten() 
    return frame_mean_score


def select_token_with_conbine_specific_retention(input_tensor, scales,video_mean_score,frame_mean_score,token_per_frame=196):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)


    combined_similarities = frame_mean_score + video_mean_score

    pruned_frames_list = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_frames_list.append(pruned_frame)

    pruned_tensor = torch.cat(pruned_frames_list, dim=0)
    return pruned_tensor
def get_indeice_of_select_token_with_conbine_specific_retention(input_tensor, scales,video_mean_score,frame_mean_score,token_per_frame=169):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)

    combined_similarities = frame_mean_score + video_mean_score

    pruned_indices = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_indices.append(sorted_indices[0])

    return pruned_indices
def convert_prune_indices(original_prune_indices, resize_h):
    global_prune_indices = []
    for frame_idx, frame_indices in enumerate(original_prune_indices):
        # 计算每个帧的起始索引
        frame_start = frame_idx * (resize_h * (resize_h + 1))
        
        # 转换帧内索引
        i = frame_indices // resize_h
        j = frame_indices % resize_h
        local_indices = i * (resize_h + 1) + j
        
        # 计算全局索引
        global_indices = frame_start + local_indices
        global_prune_indices.append(global_indices)
    
    # 合并所有帧的索引
    return torch.cat(global_prune_indices)



def get_pruned_and_retained_indices(prune_indices, resize_h):
    """
    Args:
        prune_indices: List[Tensor], 每个元素为对应帧需要prune的原始索引
        frame_number: int, 总帧数
        resize_h: int, 原始每个帧token网格的行数/列数（H）
    Returns:
        pruned_indices: Tensor, 展平后需要prune的全局索引
        retained_indices: Tensor, 展平后被保留的全局索引
    """
    frame_number=len(prune_indices)
    H = resize_h
    tokens_per_frame_original = H * H
    tokens_per_frame_new = H * (H + 1)
    
    # 计算每个帧的prune索引在展平后的全局索引
    global_prune = []
    for t in range(frame_number):
        frame_prune = prune_indices[t]
        # 确保索引在合法范围
        assert torch.all(frame_prune < tokens_per_frame_original), f"帧{t}的索引超出范围"
        # 计算帧内新索引
        i = frame_prune // H
        k_new_in_frame = frame_prune + i
        # 全局索引
        frame_offset = t * tokens_per_frame_new
        global_prune.append(frame_offset + k_new_in_frame)
    
    pruned_indices = torch.cat(global_prune)
    
    # 计算所有可能的索引并排除pruned
    total_tokens = frame_number * tokens_per_frame_new
    all_indices = torch.arange(total_tokens)
    mask = torch.ones(total_tokens, dtype=torch.bool)
    mask[pruned_indices] = False
    retained_indices = all_indices[mask]
    
    return retained_indices
    
def get_flattened_keep_indices(keep_indices, resize_h=13):
    """
    Args:
        keep_indices: List of Tensors, each of shape [num_kept_tokens], containing indices to keep for each frame.
                      Each index is in [0, resize_h*resize_h - 1].
        resize_h: int, the height/width of the resized grid (original is resize_h x resize_h).
        num_frames: int, number of frames (length of keep_indices).
    
    Returns:
        Tensor of shape [total_kept_indices] containing the indices to keep in the flattened video tensor.
    """
    H = resize_h
    W = resize_h
    W_new = W + 1  # 每行新增了一个token
    num_frames=len(keep_indices)
    
    all_indices = []
    
    for frame_idx in range(num_frames):
        # 当前帧保留的原始索引
        frame_keep = keep_indices[frame_idx]  # [num_kept]
        
        # 将原始索引转换为行和列
        rows = frame_keep // W  # 行索引，范围 [0, H-1]
        cols = frame_keep % W    # 列索引，范围 [0, W-1]
        
        # 转换到展平后的当前帧内的索引（每行有W_new个token）
        frame_local_indices = rows * W_new + cols  # [num_kept]
        
        # 转换到全局索引
        frame_start = frame_idx * (H * W_new)
        global_indices = frame_start + frame_local_indices
        all_indices.append(global_indices)
        
        # 添加当前帧每个行的新增token的索引
        # 每行的新增token在列W的位置（索引从0开始）
        row_indices = torch.arange(H, dtype=torch.long, device=frame_keep.device)  # [H]
        new_token_frame_local = row_indices * W_new + W  # [H]
        new_token_global = frame_start + new_token_frame_local  # [H]
        all_indices.append(new_token_global)
    
    # 合并所有索引并去重（理论上不会有重复，但确保安全）
    final_indices = torch.cat(all_indices, dim=0)
    return final_indices    
def select_token_with_single_specific_retention(input_tensor, scales,frame_mean_score):

    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)


    combined_similarities = frame_mean_score

    pruned_frames_list = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_frames_list.append(pruned_frame)

    pruned_tensor = torch.cat(pruned_frames_list, dim=0)
    return pruned_tensor

def select_token_with_mixed_retention(input_tensor, scales, frame_mean_score):
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)

    pruned_frames_list = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)  # 当前帧 [1, tokens_per_frame, dim]
        sim = frame_mean_score[i].unsqueeze(0)  # 当前帧的分数 [1, tokens_per_frame]
        k = k_list[i]

        # 计算五分之一和五分之四的k值
        k_large = max(1, k // 5)  # 至少保留1个token
        k_small = k - k_large

        # 获取分数最大的 k_large 个 token
        _, large_indices = torch.topk(sim, k=k_large, dim=1, largest=True, sorted=False)
        # 获取分数最小的 k_small 个 token
        _, small_indices = torch.topk(sim, k=k_small, dim=1, largest=False, sorted=False)

        # 合并索引
        combined_indices = torch.cat([large_indices, small_indices], dim=1)
        sorted_combined_indices, _ = torch.sort(combined_indices, dim=1)

        # 根据索引提取对应的 token
        pruned_frame = frame.gather(
            dim=1,
            index=sorted_combined_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_frames_list.append(pruned_frame)

    pruned_tensor = torch.cat(pruned_frames_list, dim=0)
    return pruned_tensor

def select_token_with_conbine_specific_retention_compute(input_tensor, scales):

    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)


    ############################################################################################## 
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # (num_frames, 1, dim)
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)

    video_similarities = F.cosine_similarity(frames, video_mean_token.expand_as(frames), dim=2)

    frame_similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)

    combined_similarities = video_similarities + frame_similarities

    ##################################################################################

    pruned_frames_list = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_frames_list.append(pruned_frame)

    pruned_tensor = torch.cat(pruned_frames_list, dim=0)
    return pruned_tensor



def frame_score_by_big_attn_video_mean(interpolate_features,video_mean_score):

    frame_num, token_num = interpolate_features.shape
    
    # 参数验证
    assert video_mean_score.shape == interpolate_features.shape, "video_mean_score形状必须与interpolate_features一致"
    
    # 确定每帧保留的token数量（至少保留1个）
    # k = max(1, int(token_num * 0.1))
    k = 1


    
    # 获取每帧topk的索引
    _, topk_indices = torch.topk(interpolate_features, k, dim=1,largest=True)  # (frame_num, k)
    
    # 收集对应的video_mean_score并求和
    # 使用高级索引同时获取帧索引和token索引
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]
    selected_scores = video_mean_score[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    
    return frame_scores


def frame_score_by_small_attn_video_mean(interpolate_features,video_mean_score):
    frame_num, token_num = interpolate_features.shape
    
    # 参数验证
    assert video_mean_score.shape == interpolate_features.shape, "video_mean_score形状必须与interpolate_features一致"
    
    # 确定每帧保留的token数量（至少保留1个）
    # k = max(1, int(token_num * 0.1))
    k = 1

    
    # 获取每帧topk的索引
    _, topk_indices = torch.topk(interpolate_features, k, dim=1,largest=False)  # (frame_num, k)
    
    # 收集对应的video_mean_score并求和
    # 使用高级索引同时获取帧索引和token索引
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]
    selected_scores = video_mean_score[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    
    return frame_scores




def frame_score_by_big_video_mean_attn_neg(interpolate_features,video_mean_score):

    frame_num, token_num = interpolate_features.shape
    
    # 参数验证
    assert video_mean_score.shape == interpolate_features.shape, "video_mean_score形状必须与interpolate_features一致"
    
    # 确定每帧保留的token数量（至少保留1个）
    k = max(1, int(token_num * 0.1))
    
    # 获取每帧topk的索引
    _, topk_indices = torch.topk(video_mean_score, k, dim=1,largest=True)  # (frame_num, k)
    
    # 收集对应的video_mean_score并求和
    # 使用高级索引同时获取帧索引和token索引
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=-frame_scores
    
    return frame_scores


def frame_score_by_small_video_mean_attn_neg(interpolate_features,video_mean_score):
    frame_num, token_num = interpolate_features.shape
    
    # 参数验证
    assert video_mean_score.shape == interpolate_features.shape, "video_mean_score形状必须与interpolate_features一致"
    
    # 确定每帧保留的token数量（至少保留1个）
    k = max(1, int(token_num * 0.5))
    
    # 获取每帧topk的索引
    _, topk_indices = torch.topk(video_mean_score, k, dim=1,largest=False)  # (frame_num, k)
    
    # 收集对应的video_mean_score并求和
    # 使用高级索引同时获取帧索引和token索引
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)

    frame_scores=-frame_scores
    
    return frame_scores



def compute_frame_score_top_10_percent_interpolate_features_positive(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    
    k = max(1, int(token_num * 0.1))

    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)

    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)

    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    
    return frame_scores



def compute_frame_score_top_10_percent_interpolate_features_negative(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    k = max(1, int(token_num * 0.1))
    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=-frame_scores
    return frame_scores

def compute_frame_score_top_10_percent_interpolate_key_negative(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    k = max(1, int(token_num * 0.1))
    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=-frame_scores
    return frame_scores


def compute_frame_score_top_10_percent_interpolate_key_positive(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    k = max(1, int(token_num * 0.1))
    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=frame_scores
    return frame_scores

def compute_frame_score_full_interpolate_key_negative(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    k = max(1, int(token_num * 1))
    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=-frame_scores
    return frame_scores


def compute_frame_score_full_interpolate_key_positive(interpolate_features):

    frame_num, token_num = interpolate_features.shape
    k = max(1, int(token_num * 1))
    _, topk_indices = torch.topk(interpolate_features, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=interpolate_features.device)[:, None]  # (frame_num, 1)
    selected_scores = interpolate_features[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=frame_scores
    return frame_scores

def compute_frame_score_with_above_mean_tokens(interpolate_features):

    # 计算每帧的平均值
    frame_means = interpolate_features.mean(dim=1, keepdim=True)  # 形状 [frame_number, 1]
    
    # 找出大于每帧平均值的 token 的布尔掩码
    above_mean_mask = interpolate_features > 2*frame_means  # 形状 [frame_number, token_number]
    
    # 使用掩码筛选大于平均值的 token
    above_mean_scores = interpolate_features * above_mean_mask  # 将小于等于平均值的 token 置为 0
    
    # 统计每帧中大于平均值的 token 数量
    above_mean_counts = above_mean_mask.sum(dim=1)  # 形状 [frame_number]
        
    # 计算每帧中大于平均值的 token 的分数的平均值

    frame_scores = above_mean_scores.sum(dim=1)   # 形状 [frame_number]
    # frame_scores = above_mean_scores.sum(dim=1)
    frame_scores=-frame_scores

    return frame_scores

def compute_frame_score_top_10_percent_video_mean_score_positive(video_mean_score):

    frame_num, token_num = video_mean_score.shape
    
    k = max(1, int(token_num * 0.1))

    _, topk_indices = torch.topk(video_mean_score, k, dim=1, largest=True, sorted=False)  # (frame_num, k)

    frame_indices = torch.arange(frame_num, device=video_mean_score.device)[:, None]  # (frame_num, 1)
    selected_scores = video_mean_score[frame_indices, topk_indices]  # (frame_num, k)

    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    
    return frame_scores



def compute_frame_score_top_10_percent_video_mean_score_negative(video_mean_score):

    frame_num, token_num = video_mean_score.shape
    k = max(1, int(token_num * 0.1))
    _, topk_indices = torch.topk(video_mean_score, k, dim=1, largest=True, sorted=False)  # (frame_num, k)
    frame_indices = torch.arange(frame_num, device=video_mean_score.device)[:, None]  # (frame_num, 1)
    selected_scores = video_mean_score[frame_indices, topk_indices]  # (frame_num, k)
    frame_scores = selected_scores.sum(dim=1)  # (frame_num,)
    frame_scores=-frame_scores
    return frame_scores





def compute_frame_score_with_above_mean_tokens_video_mean_score(input_tensor):
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    avg_per_frame = frames.mean(dim=1, keepdim=True)  # (num_frames, 1, dim)
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)
        # 计算余弦相似度
    frame_scores = F.cosine_similarity(avg_per_frame, video_mean_token.expand_as(avg_per_frame), dim=2)

    # 调整形状为 [frame_number]
    frame_scores = frame_scores.squeeze()  # 去掉多余的维度
    frame_scores=-frame_scores

    return frame_scores



def random_drop_tokens(tensor, drop_ratio=0.75):
    """
    随机丢弃指定比例的 token。

    参数:
        tensor (torch.Tensor): 输入 Tensor，形状为 [token_number, channel]。
        drop_ratio (float): 丢弃的比例，默认为 0.75。

    返回:
        torch.Tensor: 保留的 token 组成的新 Tensor。
    """
    # 获取 token 数量
    token_number = tensor.size(0)
    
    # 计算需要保留的 token 数量
    keep_ratio = 1 - drop_ratio
    keep_number = int(token_number * keep_ratio)
    
    # 随机选择保留的 token 索引
    indices = torch.randperm(token_number)[:keep_number]
    
    # 按索引提取保留的 token
    result = tensor[indices]
    
    return result





def get_indeice_of_select_token_with_single_specific_retention(input_tensor, scales,video_mean_score,token_per_frame=169):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)

    combined_similarities = video_mean_score

    pruned_indices = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_indices.append(sorted_indices[0])

    return pruned_indices



def get_indeice_of_select_token_with_subtract_conbine_specific_retention(input_tensor, scales,video_mean_score,frame_mean_score,token_per_frame=169):

    num_frame = input_tensor.shape[0] // token_per_frame
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    total_frames, tokens_per_frame, dim = frames.shape

    assert len(scales) == num_frame, "scales长度必须等于帧数"
    scales = torch.as_tensor(scales, dtype=torch.float32, device=frames.device)
    k_list = []
    for scale in scales:
        scaled_value = scale.item() * tokens_per_frame  # 转换为Python标量
        k = max(1, int(round(scaled_value)))
        k_list.append(k)

    combined_similarities = frame_mean_score - video_mean_score

    pruned_indices = []
    for i in range(num_frame):
        frame = frames[i].unsqueeze(0)
        sim = combined_similarities[i].unsqueeze(0)
        k = k_list[i]

        _, indices = torch.topk(sim, k=k, dim=1, largest=False, sorted=False)
        sorted_indices, _ = torch.sort(indices, dim=1)

        pruned_frame = frame.gather(
            dim=1,
            index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        ).view(-1, dim)
        pruned_indices.append(sorted_indices[0])

    return pruned_indices