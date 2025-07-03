import torch
import numpy as np
import torch.nn.functional as F


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

def generate_scales_from_frame_video_score(frame_scores, base_scale=0.3, temperature=0.1):
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

def compute_video_mean_score(input_tensor):
    """计算每个token的余弦相似度分数"""
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, dim)
    token_scores = F.cosine_similarity(frames, avg_token.expand_as(frames), dim=2)
    # video_mean_score=token_scores.flatten() 
    return token_scores

def compute_video_mean_score_high_var(input_tensor):
    """计算每个token在方差最大前一半通道上的余弦相似度分数"""
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // 196
    frames = selected_tensor.view(num_frame, 196, -1)  # [num_frame, 196, k]
    
    # 在筛选后的通道上计算均值
    avg_token = frames.mean(dim=(0, 1), keepdim=True)  # [1, 1, k]
    
    # 计算余弦相似度
    token_scores = F.cosine_similarity(frames, avg_token.expand_as(frames), dim=2)
    
    return token_scores

def compute_frame_mean_score_high_var(input_tensor):
    """计算每个token在方差最大前一半通道上的余弦相似度分数"""
    # 计算每个通道的方差 [dim]
    channel_var = input_tensor.var(dim=0, unbiased=False)  # 使用有偏方差保持一致性
    
    # 选择方差最大的前一半通道
    k = channel_var.shape[0] // 2
    _, topk_indices = torch.topk(channel_var, k=k,largest=False)
    
    # 筛选高方差通道
    selected_tensor = input_tensor[:, topk_indices]  # [N, k]
    
    # 重塑为视频帧结构
    num_frame = selected_tensor.shape[0] // 196
    frames = selected_tensor.view(num_frame, 196, -1)  # [num_frame, 196, k]
    
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


def select_token_with_conbine_specific_retention(input_tensor, scales,video_mean_score,frame_mean_score):

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