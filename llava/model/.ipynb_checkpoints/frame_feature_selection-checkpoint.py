import torch
import numpy as np
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment
def dycole_ttm(image_feature, num_tokens_per_frame = 196):
    # Split frames into tokens
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    # Calculate similarities between adjacent even frames
    similarities = []
    mean_similarities=[]
    frame_number=image_feature.shape[0]//196

    for i in range(0, num_frames - 1, 1):
        # Get tokens for adjacent frames
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]

        # Calculate cosine similarity between normalized tokens
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity.max().item())
        mean_similarities.append(similarity.mean().item())
    mean_similarities_tensor = torch.tensor(mean_similarities)
    min_val = mean_similarities_tensor.min()
    max_val = mean_similarities_tensor.max()

    # 处理常数情况（所有值相同）
    if max_val - min_val < 1e-6:  # 防止除以0
        normalized_means = torch.zeros_like(mean_similarities_tensor)
    else:
        normalized_means = (mean_similarities_tensor - min_val) / (max_val - min_val)

    # 转换回列表（如果需要）
    normalized_means = normalized_means.tolist()
    for idx in range(1, len(normalized_means), 2):
        normalized_means[idx] += 0.5
    
    sorted_indices = np.argsort(normalized_means)
    keep_indices = sorted_indices[len(sorted_indices)//2:]+1  # 保留后半部分（高分部分）
    selected_frames = [i for i in keep_indices]
    
    selected_features = image_feature.view(frame_number,196,896)[selected_frames, :, :].view(-1,896)

    return selected_features


def dycoke_ttm(image_feature, num_tokens_per_frame = 196, merging_ratio = 0.75):
    # Split frames into tokens
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio
    # Calculate similarities between adjacent even frames
    similarities = []
    mean_similarities=[]

    for i in range(0, num_frames - 1, 1):
        # Get tokens for adjacent frames
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]

        # Calculate cosine similarity between normalized tokens
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
 
    similarities = torch.stack([torch.tensor(similarity) for similarity in similarities])

    # Process even frames
    modified_image_feature = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
        
        avg_similarity = similarities[(i - 2) // 2]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(frame2_tokens[tokens_to_keep])


    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        similarity = torch.nn.functional.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    odd_similarities = torch.stack([torch.tensor(similarity) for similarity in odd_similarities])

    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        avg_similarity = odd_similarities[i // 4]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature[i] = frame1_tokens
        modified_image_feature[i + 2] = frame2_tokens[tokens_to_keep]

    # Combine all tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    return combined_tokens
def frame_wise_ttm(image_feature,threshold=0.9):

    frame_token_length=196
    all_image_token_length=image_feature.shape[0]
    frame_number=all_image_token_length//frame_token_length

    # Calculate similarities between adjacent even frames
    base_frames_index = [0]   
    L1_score=[]
    L1_distance=[]
    current_base = image_feature[0 : frame_token_length]
    for i in range(0, frame_number, 1):
        # Get tokens for adjacent frames
        frame1_tokens = current_base
        frame2_tokens = image_feature[(i) * frame_token_length: (i+1) * frame_token_length]
        # Calculate cosine similarity between normalized tokens
        # frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        # frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        l1_distance = torch.abs(frame1_tokens - frame2_tokens)
        l1_score = l1_distance.mean().item()
        if l1_score > threshold:

            base_frames_index.append(i)       # 记录新 clip 的 base frame 索引
            current_base = frame2_tokens
            L1_score.append(l1_score)
            L1_distance.append(l1_distance)
        else:
            # 若距离未超过阈值，则当前帧属于当前 clip
            L1_score.append(l1_score)
            L1_distance.append(l1_distance)    
    return base_frames_index, L1_score,L1_distance

def generate_scales_whith_score(base_frames_index,L1_score,base_scale=0.3,temperature=0.1):
    # Normalize the scores and apply softmax
    frame_number=len(L1_score)
    shifted_scores = (L1_score - np.max(L1_score)) / temperature
    exp_scores = np.exp(shifted_scores)
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
    # Calculate scales ensuring no scale exceeds 1
    if np.sum(softmax_scores) == 0:
        scales = [base_scale] * frame_number
    else:
        scales = base_scale * (1 + softmax_scores - np.mean(softmax_scores))
        scales = np.clip(scales, None, 1.0)
    final_scales = []
    scale_idx = 0

    for i in range(frame_number):
        if i in base_frames_index:
            final_scales.append(1.0)
        else:
            final_scales.append(scales[scale_idx])
            scale_idx += 1
    return  final_scales


def select_topk_token_whith_L1_distance(image_feature,l1_distance,base_frames_index,scales):
    frame_token_length = 196

    # 对于每个非 base frame，根据对应的 retention_ratio 计算需要保留的 token 数量，并用 topk 选取
    num_frames = image_feature.shape[0]//196
    keep_indices = [[] for _ in range(num_frames)]  # 初始化空容器
    modified_image_feature=[[] for _ in range(num_frames)]
    # 处理基准帧（保留全部token）
    # for frame in base_frames_index:
    #     # keep_indices[frame] = torch.arange(frame_token_length)
    #     random_indices = torch.randperm(frame_token_length)
    #     half_length = frame_token_length // 3
    #     keep_indices[frame] = random_indices[:half_length]
    #     base_frame_tokens = image_feature[(frame) * frame_token_length: (frame+1) * frame_token_length]
    #     modified_image_feature[frame] = base_frame_tokens[keep_indices[frame],:]
     # 处理基准帧（保留部分token）
    base_frame_ratio=0.3
    for frame in base_frames_index:
        top_k = int(round(base_frame_ratio * frame_token_length))
        if top_k == 0:  # 不保留任何token
            keep_indices[frame] = []
        else:
            # 均匀采样 top_k 个 token
            step = frame_token_length // top_k
            keep_indices[frame] = list(range(0, frame_token_length, step))[:top_k]
            
            # 剩下的 token
            remaining_indices = [i for i in range(frame_token_length) if i not in keep_indices[frame]]
            
            # 计算两组之间的 cosine similarity
            base_frame_tokens = image_feature[(frame) * frame_token_length: (frame + 1) * frame_token_length]
            sampled_tokens = base_frame_tokens[keep_indices[frame]]
            remaining_tokens = base_frame_tokens[remaining_indices]
            
            sampled_tokens_np = sampled_tokens.detach().cpu().numpy()
            remaining_tokens_np = remaining_tokens.detach().cpu().numpy()
            
            cos_sim = F.cosine_similarity(remaining_tokens.unsqueeze(1), sampled_tokens.unsqueeze(0), dim=2)
            
            # 找到与 sampled_tokens 最相似的 remaining_tokens
            max_sim_indices = cos_sim.argmax(axis=1)
            selected_remaining_indices = [remaining_indices[i] for i in max_sim_indices]
            
            # 合并两组 token
            keep_indices[frame].extend(selected_remaining_indices)
            keep_indices[frame] = list(set(keep_indices[frame]))  # 去重
            
        base_frame_tokens = image_feature[(frame) * frame_token_length: (frame + 1) * frame_token_length]
        modified_image_feature[frame] = base_frame_tokens[keep_indices[frame], :]

    # 处理非基准帧
    non_base = [i for i in range(num_frames) if i not in base_frames_index]

    for i, frame in enumerate(non_base):
        
        ratio = scales[i]
        top_k = int(round(ratio * frame_token_length))
        if top_k == 0:  # 不保留任何token
            keep_indices[frame] = []
        else:  # 按得分选择top_k
            frame_scores = l1_distance[frame].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame]=top_indices

        base_frame_tokens = image_feature[(frame) * frame_token_length: (frame+1) * frame_token_length]
        modified_image_feature[frame] = base_frame_tokens[keep_indices[frame]]
    # 将结果转换为numpy数组列表
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    return combined_tokens


def local_compression(tensor, ratio):
    """
    Args:
        tensor: Input tensor of shape (3136, 896)
        ratio: Pruning ratio (0.0-1.0), ratio of tokens to prune
    Returns:
        Pruned tensor of shape (N, 896), where N = 3136*(1-ratio)
    """
    # Reshape to (16, 14, 14, 896)
    frame_number=tensor.shape[0]//196

    x = tensor.view(frame_number, 14, 14, 896)
    
    # Unfold into 2x2 patches
    patches = x.unfold(1, 2, 2).unfold(2, 2, 2)  # Shape: (16, 7, 7, 2, 2, 896)
    
    # Reshape to (num_patches, patch_size, features)
    patches = patches.contiguous().view(-1, 4, 896)  # 16*7*7=784 patches, 4 tokens per patch
    
    # Normalize features for cosine similarity calculation
    normalized = F.normalize(patches, p=2, dim=-1)
    
    # Compute cosine similarity matrices for all patches
    sim_matrix = torch.matmul(normalized, normalized.transpose(1, 2))  # Shape: (784, 4, 4)
    
    # Calculate scores for each token
    scores = (sim_matrix.sum(dim=2) - 1) / 3  # Subtract self-similarity (1) and average over 3 neighbors
    scores = scores.view(-1)  # Flatten to (3136,)
    
    # Create pruning mask
    num_to_prune = int(ratio * scores.numel())
    _, prune_indices = torch.topk(scores, k=num_to_prune, largest=True)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[prune_indices] = False
    
    # Apply pruning mask to original tensor
    pruned_tensor = tensor[mask]
    
    return pruned_tensor



def prune_features(image_features, attention_scores, num_frames, tokens_per_frame=196, prune_ratio=0.7):
    """
    image_features: 原始特征矩阵，形状为 (总token数, 特征维度) = (3136, 896)
    attention_scores: 注意力分数矩阵，形状需与image_features一致 (3136,)
    num_frames: 视频帧数 (默认16)
    tokens_per_frame: 每帧token数 (默认196)
    prune_ratio: 剪枝比例 (默认0.5)
    """
    # 重塑为帧优先的维度 (帧数, token数, 特征维度)
    image_features_reshaped = image_features.view(num_frames, tokens_per_frame, -1)
    attention_scores_reshaped = attention_scores.view(num_frames, tokens_per_frame)
    
    pruned_list = []
    
    for frame_idx in range(num_frames):
        # 获取当前帧的feature和score
        frame_features = image_features_reshaped[frame_idx]  # (196, 896)
        frame_scores = attention_scores_reshaped[frame_idx]  # (196,)
        
        # 计算需要保留的token数量
        keep_num = int(tokens_per_frame * (1 - prune_ratio))
        
        # 获取分数最低的索引（ascending排序取前keep_num个）
        _, keep_indices = torch.topk(frame_scores, k=keep_num)  # PyTorch版本
        # 若用NumPy：
        # keep_indices = np.argpartition(frame_scores, keep_num)[:keep_num]
        
        # 保持原始顺序排序索引
        keep_indices_sorted, _ = torch.sort(keep_indices)
        # 若用NumPy：
        # keep_indices_sorted = np.sort(keep_indices)
        
        # 保留对应特征
        pruned_frame = frame_features[keep_indices_sorted]
        pruned_list.append(pruned_frame)
    
    # 拼接所有帧的结果
    pruned_features = torch.cat(pruned_list, dim=0)  # PyTorch版本
    return pruned_features

def prune_tokens(input_tensor, prune_ratio):

 
    # 调整张量形状为 (16帧, 196 tokens/帧, 896维)
    num_frame=input_tensor.shape[0]//196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧需要保留的token数量（四舍五入）
    k = int(round((1 - prune_ratio) * tokens_per_frame))
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (16, 1, 896)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 获取每帧的topk最小相似度的索引
    _, indices = torch.topk(similarities, k=k, dim=1, largest=True, sorted=False)
    
    # 对索引排序以保持原始token顺序
    sorted_indices, _ = torch.sort(indices, dim=1)
    
    # 收集保留的token
    pruned_frames = frames.gather(
        dim=1,
        index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
    )
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.view(-1, dim)
    
    return pruned_tensor



def select_least_similar_frames(image_feature: torch.Tensor) -> torch.Tensor:
    """
    通过余弦相似度筛选与视频整体差异最大的帧
    保留相似度最低的一半帧以减少冗余计算
    
    Args:
        image_feature (torch.Tensor): 原始视频特征，形状应为 (N_frames*196, 896)
        
    Returns:
        torch.Tensor: 筛选后的视频特征，形状为 (K*196, 896)，其中 K = N_frames//2
    """
    num_frames = image_feature.shape[0] // 196  # 计算总帧数
    input_tensor = image_feature

    # 计算全局视频特征均值
    video_avg = input_tensor.mean(dim=0).view(1, 1, -1)
    
    # 重塑为帧结构 (frames, tokens, features)
    frames = input_tensor.view(num_frames, 196, -1)
    
    # 计算每帧均值特征
    frame_avg = frames.mean(dim=1).unsqueeze(1)
    
    # 计算与全局特征的余弦相似度
    cos_sim = F.cosine_similarity(
        frame_avg,
        video_avg.expand_as(frame_avg),
        dim=2
    ).squeeze()
    
    # 选择相似度最低的k帧
    k = num_frames // 2
    _, indices = torch.topk(cos_sim, k=k, largest=False)
    
    # 重组特征并保持原始结构
    selected_frames = frames[indices]
    return selected_frames.view(-1, image_feature.size(-1))
def select_most_similar_frames(image_feature: torch.Tensor) -> torch.Tensor:
    """
    通过余弦相似度筛选与视频整体差异最大的帧
    保留相似度最低的一半帧以减少冗余计算
    
    Args:
        image_feature (torch.Tensor): 原始视频特征，形状应为 (N_frames*196, 896)
        
    Returns:
        torch.Tensor: 筛选后的视频特征，形状为 (K*196, 896)，其中 K = N_frames//2
    """
    num_frames = image_feature.shape[0] // 196  # 计算总帧数
    input_tensor = image_feature

    # 计算全局视频特征均值
    video_avg = input_tensor.mean(dim=0).view(1, 1, -1)
    
    # 重塑为帧结构 (frames, tokens, features)
    frames = input_tensor.view(num_frames, 196, -1)
    
    # 计算每帧均值特征
    frame_avg = frames.mean(dim=1).unsqueeze(1)
    
    # 计算与全局特征的余弦相似度
    cos_sim = F.cosine_similarity(
        frame_avg,
        video_avg.expand_as(frame_avg),
        dim=2
    ).squeeze()
    
    # 选择相似度最低的k帧
    k = num_frames // 2
    _, indices = torch.topk(cos_sim, k=k, largest=False)
    
    # 重组特征并保持原始结构
    selected_frames = frames[indices]
    return selected_frames.view(-1, image_feature.size(-1))

    
def select_frames_by_small_attention_similarity(
    image_feature: torch.Tensor,  # 输入特征 (3136, 896)
    attn_map: torch.Tensor        # 注意力分数 (16, 196, 196)
) -> torch.Tensor:
    # 计算全局注意力均值 (1, 1, 196)
    frame_number=image_feature.size(0)//196
    video_attn_avg = attn_map.mean(dim=(0, 1))  # 跨帧和token维度求平均
    video_attn_avg = video_attn_avg.view(1, 1, -1)

    # 计算每帧注意力均值 (16, 1, 196)
    frame_attn_avg = attn_map.mean(dim=1)       # 每帧token维度平均
    frame_attn_avg = frame_attn_avg.unsqueeze(1)

    # 计算余弦相似度 (16,)
    cos_sim = F.cosine_similarity(
        frame_attn_avg.expand(-1, 1, 196),  # 扩展维度 (16,1,1,196)
        video_attn_avg.expand(frame_number, 1, 196),  # 扩展维度 (16,1,1,196)
        dim=-1
    ).squeeze()

    # 筛选相似度最低的帧
    k = attn_map.size(0) // 2
    _, indices = torch.topk(cos_sim, k=k, largest=False)

    # 重组视频特征
    num_frames = image_feature.size(0) // 196
    frames = image_feature.view(num_frames, 196, -1)
    return frames[indices].view(-1, image_feature.size(-1))

def select_frames_by_big_attention_similarity(
    image_feature: torch.Tensor,  # 输入特征 (3136, 896)
    attn_map: torch.Tensor        # 注意力分数 (16, 196, 196)
) -> torch.Tensor:
    # 计算全局注意力均值 (1, 1, 196)
    video_attn_avg = attn_map.mean(dim=(0, 1))  # 跨帧和token维度求平均
    video_attn_avg = video_attn_avg.view(1, 1, -1)

    # 计算每帧注意力均值 (16,  1, 196)
    frame_attn_avg = attn_map.mean(dim=1)       # 每帧token维度平均
    frame_attn_avg = frame_attn_avg.unsqueeze(1)

    # 计算余弦相似度 (16,)
    num_frames = image_feature.size(0) // 196
    cos_sim = F.cosine_similarity(
        frame_attn_avg.expand(-1, 1, 196),  # 扩展维度 (16,1,1,196)
        video_attn_avg.expand(num_frames, 1, 196),  # 扩展维度 (16,1,1,196)
        dim=-1
    ).squeeze()

    # 筛选相似度最低的帧
    k = attn_map.size(0) // 2
    _, indices = torch.topk(cos_sim, k=k, largest=True)

    # 重组视频特征
    num_frames = image_feature.size(0) // 196
    frames = image_feature.view(num_frames, 196, -1)
    return frames[indices].view(-1, image_feature.size(-1))


def select_token_whith_big_importance(image_feature, attn_map):
    # Reshape image features to (16, 196, 896)
    num_frames=image_feature.size(0)//196
    image_reshaped = image_feature.view(num_frames, 196, -1)
    
    pruned_features = []
    
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Get attention scores for current frame (196, 196)
        frame_attn = attn_map[frame_idx]
        
        # Calculate importance scores: sum of attention received from all queries
        # shape: (196,)
        token_scores = torch.sum(frame_attn, dim=-1)
        
        # Select indices of least important tokens (smallest 50%)
        _, keep_indices = torch.topk(token_scores, 
                                   k=num_frames//2,  # 196 // 2
                                   largest=True)
        
        # Gather selected features (98, 896)
        pruned_frame = image_reshaped[frame_idx, keep_indices, :]
        pruned_features.append(pruned_frame)
    
    # Concatenate all frames and return
    return torch.cat(pruned_features, dim=0)

def select_token_whith_small_importance(image_feature, attn_map):
    # Reshape image features to (16, 196, 896)
    num_frames=image_feature.size(0)//196
    image_reshaped = image_feature.view(num_frames, 196, -1)
    
    pruned_features = []
    
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Get attention scores for current frame (196, 196)
        frame_attn = attn_map[frame_idx]
        
        # Calculate importance scores: sum of attention received from all queries
        # shape: (196,)
        token_scores = torch.sum(frame_attn, dim=-1)
        
        # Select indices of least important tokens (smallest 50%)
        _, keep_indices = torch.topk(token_scores, 
                                   k=num_frames//2,  # 196 // 2
                                   largest=False)
        
        # Gather selected features (98, 896)
        pruned_frame = image_reshaped[frame_idx, keep_indices, :]
        pruned_features.append(pruned_frame)
    
    # Concatenate all frames and return
    return torch.cat(pruned_features, dim=0)

def select_token_by_big_cosine_similarity_with_video_mean_token(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, 896维)
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧需要保留的token数量（四舍五入）
    k = int(round((1 - 0.5) * tokens_per_frame))
    
    # 计算整个视频的全局平均token
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)
    
    # 计算每个token与全局平均token的余弦相似度
    similarities = F.cosine_similarity(
        frames,
        video_mean_token.expand_as(frames),  # 将全局平均扩展为与frames相同的形状
        dim=2  # 在896维度计算相似度
    )
    
    # 获取每帧的topk最大相似度的索引
    _, indices = torch.topk(similarities, k=k, dim=1, largest=True, sorted=False)
    
    # 对索引排序以保持原始token顺序
    sorted_indices, _ = torch.sort(indices, dim=1)
    
    # 收集保留的token
    pruned_frames = frames.gather(
        dim=1,
        index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
    )
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.view(-1, dim)
    
    return pruned_tensor

def select_token_by_small_cosine_similarity_with_video_mean_token(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, 896维)
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧需要保留的token数量（四舍五入）
    k = int(round((1 - 0.5) * tokens_per_frame))
    
    # 计算整个视频的全局平均token
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)
    
    # 计算每个token与全局平均token的余弦相似度
    similarities = F.cosine_similarity(
        frames,
        video_mean_token.expand_as(frames),  # 将全局平均扩展为与frames相同的形状
        dim=2  # 在896维度计算相似度
    )
    
    # 获取每帧的topk最大相似度的索引
    _, indices = torch.topk(similarities, k=k, dim=1, largest=False, sorted=False)
    
    # 对索引排序以保持原始token顺序
    sorted_indices, _ = torch.sort(indices, dim=1)
    
    # 收集保留的token
    pruned_frames = frames.gather(
        dim=1,
        index=sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
    )
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.view(-1, dim)
    
    return pruned_tensor

def select_frame_by_odd(feature_tensor):
    tokens_per_frame=196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame

    pruned = feature_tensor.view(n_frames, tokens_per_frame, -1)
    pruned = pruned[1::2]

    return pruned.reshape(-1, feature_tensor.size(-1))

def select_frame_by_even(feature_tensor):
    tokens_per_frame=196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame

    pruned = feature_tensor.view(n_frames, tokens_per_frame, -1)
    pruned = pruned[::2] 

    return pruned.reshape(-1, feature_tensor.size(-1))

def select_frame_by_big_before_adjacent_L1(image_feature: torch.Tensor) -> torch.Tensor:

    # 自动计算帧数（每帧包含196个token）
    num_frames = image_feature.size(0) // 196
    image_features = image_feature.view(num_frames, 196, -1)

    # 计算每帧与前帧的L1距离（从第1帧开始）
    frame_distances = []
    for i in range(1, num_frames):
        prev = image_features[i-1]
        curr = image_features[i]
        frame_distances.append( (i, torch.abs(curr - prev).sum().item()) )

    # 剪枝策略：选择变化最大的前50%帧（向下取整）
    num_prune = num_frames // 2
    sorted_distances = sorted(frame_distances, key=lambda x: x[1], reverse=True)
    pruned_indices = {idx for idx, _ in sorted_distances[:num_prune]}

    # 保留未剪枝的帧（始终包含第0帧）
    keep_indices = [i for i in range(num_frames) if i == 0 or i not in pruned_indices]
    
    # 恢复形状并返回
    return image_features[keep_indices].view(-1, image_feature.size(1))

def select_frame_by_small_before_adjacent_L1(image_feature: torch.Tensor) -> torch.Tensor:

    # 自动计算帧数（每帧包含196个token）
    num_frames = image_feature.size(0) // 196
    image_features = image_feature.view(num_frames, 196, -1)

    # 计算每帧与前帧的L1距离（从第1帧开始）
    frame_distances = []
    for i in range(1, num_frames):
        prev = image_features[i-1]
        curr = image_features[i]
        frame_distances.append( (i, torch.abs(curr - prev).sum().item()) )

    # 剪枝策略：选择变化最大的前50%帧（向下取整）
    num_prune = num_frames // 2
    sorted_distances = sorted(frame_distances, key=lambda x: x[1], reverse=False)
    pruned_indices = {idx for idx, _ in sorted_distances[:num_prune]}

    # 保留未剪枝的帧（始终包含第0帧）
    keep_indices = [i for i in range(num_frames) if i == 0 or i not in pruned_indices]
    
    # 恢复形状并返回
    return image_features[keep_indices].view(-1, image_feature.size(1))


# def select_frame_by_big_neighbou_L1(image_feature):
#     # 将输入tensor重塑为(16, 196, 896)
#     frame_number=image_feature.size(0)//196
#     frames = image_feature.view(frame_number, 196, 896)
    
#     # 计算相邻帧之间的L1距离（使用总和）
#     current_frames = frames[:-1]  # 前15帧
#     next_frames = frames[1:]      # 后15帧
#     adjacent_distances = torch.sum(torch.abs(current_frames - next_frames), dim=(1, 2))  # 形状(15,)
    
#     # 计算每个帧的评分（边界帧使用单边距离，中间帧使用双边距离之和）
#     scores = torch.zeros(frame_number, device=image_feature.device)
#     scores[0] = adjacent_distances[0]          # 第一帧
#     scores[-1] = adjacent_distances[-1]        # 最后一帧
#     scores[1:-1] = adjacent_distances[:-1] + adjacent_distances[1:]  # 中间帧
    
#     # 选择需要剪枝的帧（评分最高的50%）
#     k = frames.size(0) // 2
#     _, indices_to_prune = torch.topk(scores, k=k, largest=True)

#     mask = torch.ones(16, dtype=torch.bool, device=image_feature.device)
#     mask[indices_to_prune] = False
#     pruned_frames = frames[mask]

#     return pruned_frames.reshape(-1, 896)