import torch
import numpy as np
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment
def dycoke_ttm_retention_llava_video(image_feature, num_tokens_per_frame=169, retention_ratio=0.07):
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    device = image_feature.device
    
    # 计算相邻帧之间的相似度
    similarities = []
    for i in range(num_frames - 1):
        frame1 = image_feature[i*num_tokens_per_frame : (i+1)*num_tokens_per_frame]
        frame2 = image_feature[(i+1)*num_tokens_per_frame : (i+2)*num_tokens_per_frame]
        frame1_norm = torch.nn.functional.normalize(frame1, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
    similarities = torch.stack(similarities)  # (num_frames-1, num_tokens)

    # 初始化保留索引列表（每帧保留所有token）
    retained_indices = [torch.arange(num_tokens_per_frame, device=device) for _ in range(num_frames)]

    # 处理偶数帧对
    for i in range(0, num_frames-1, 2):
        avg_similarity = similarities[i]  # 修正索引计算
        num_keep = int(retention_ratio * num_tokens_per_frame)
        keep_indices = avg_similarity.topk(num_keep, largest=False).indices
        retained_indices[i+1] = keep_indices  # 奇数帧保留部分token

    # 计算间隔帧相似度
    odd_similarities = []
    for i in range(0, num_frames-4, 4):
        frame1 = image_feature[i*num_tokens_per_frame : (i+1)*num_tokens_per_frame]
        frame2 = image_feature[(i+2)*num_tokens_per_frame : (i+3)*num_tokens_per_frame]
        similarity = torch.nn.functional.cosine_similarity(frame1, frame2, dim=1)
        odd_similarities.append(similarity)
    if odd_similarities:
        odd_similarities = torch.stack(odd_similarities)

    # 处理间隔帧
    for idx, i in enumerate(range(0, num_frames-4, 4)):
        avg_similarity = odd_similarities[idx]
        num_keep = int(retention_ratio * num_tokens_per_frame)
        keep_indices = avg_similarity.topk(num_keep, largest=False).indices
        retained_indices[i+2] = keep_indices  # 间隔帧保留部分token


    

    return retained_indices




def dycoke_ttm_retention_llava_ov(image_feature, num_tokens_per_frame = 196, retention_ratio = 0.07):
    
    # Split frames into tokens
    num_frames = image_feature.shape[0] // num_tokens_per_frame

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
        # num_tokens_to_keep = int(retention_ratio * num_tokens_per_frame)
        num_tokens_to_keep=1

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
        # num_tokens_to_keep = int(retention_ratio * num_tokens_per_frame)
        num_tokens_to_keep=1

        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature[i] = frame1_tokens
        modified_image_feature[i + 2] = frame2_tokens[tokens_to_keep]

    # Combine all tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)

    return combined_tokens


def select_base_frames(image_feature,threshold=0.9):

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
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)  
    # add a small constant to avoid division by zero
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


def generate_scales_whith_mean_frame_score(input_tensor,base_scale=0.4,temperature=0.1):
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, dim)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)
    
    # 计算每帧的总得分（相似度之和）
    frame_scores = similarities.sum(dim=1)  # 形状 (num_frames,)
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
def select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio):
    frame_token_length = 196
    num_frames = image_feature.shape[0] // frame_token_length
    keep_indices = [[] for _ in range(num_frames)]
    modified_image_feature = [[] for _ in range(num_frames)]

    # 处理基准帧
    for frame in base_frames_index:
        # 计算需要保留的token数量
        top_k = int(round(base_frame_retention_ratio * frame_token_length))
        
        if top_k == 0:
            keep_indices[frame] = []
        else:
            # 获取当前帧的所有token
            base_frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            
            # 计算该帧的平均token
            mean_token = base_frame_tokens.mean(dim=0, keepdim=True)
            
            # 计算每个token与平均token的余弦相似度
            similarities = F.cosine_similarity(base_frame_tokens, mean_token, dim=1)
            
            # 选择相似度最小的top_k个token
            _, top_indices = torch.topk(similarities, k=top_k, largest=False)
            
            # 保存索引并确保顺序
            sorted_indices, _ = torch.sort(top_indices)
            keep_indices[frame] = sorted_indices.tolist()
        
        # 提取保留的token
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 处理非基准帧（保持原有逻辑不变）
    non_base = [i for i in range(num_frames) if i not in base_frames_index]
    for i, frame in enumerate(non_base):
        ratio = scales[i]
        top_k = int(round(ratio * frame_token_length))
        if top_k == 0:
            keep_indices[frame] = []
        else:
            frame_scores = L1_distance[frame].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame] = top_indices.tolist()
        
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 合并所有保留的token
    valid_tokens = [tokens for tokens in modified_image_feature if len(tokens) > 0]
    combined_tokens = torch.cat(valid_tokens, dim=0) if valid_tokens else torch.tensor([])
    return combined_tokens

def random_retain_tokens(tensor: torch.Tensor, ratio: float = 0.15) -> torch.Tensor:

    # 参数有效性检查
    if not 0 <= ratio <= 1:
        raise ValueError(f"无效的ratio值: {ratio}，应在[0,1]区间内")
    
    # 计算需要保留的token数量
    token_number = tensor.size(0)
    k = int(token_number * ratio)
    selected_indices = torch.randperm(token_number)[:k]
    return tensor[selected_indices]

def full_tokens(tensor):
    return tensor
def select_token_base_by_video_mean_then_random_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio):
    frame_token_length = 196
    num_frames = image_feature.shape[0] // frame_token_length
    keep_indices = [[] for _ in range(num_frames)]
    modified_image_feature = [[] for _ in range(num_frames)]

    # 处理基准帧
    for frame in base_frames_index:
        # 第一阶段：压缩到50%
        initial_retention_ratio = 0.5
        top_k_initial = int(round(initial_retention_ratio * frame_token_length))
        
        # 第二阶段：计算最终需要保留的数量
        top_k_final = int(round(base_frame_retention_ratio * frame_token_length))
        num_desired = min(top_k_final, top_k_initial)  # 确保不超过初始压缩量

        if num_desired <= 0:
            keep_indices[frame] = []
        else:
            # 获取当前帧的所有token
            base_frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            
            # 计算该帧的平均token
            mean_token = base_frame_tokens.mean(dim=0, keepdim=True)
            
            # 计算每个token与平均token的余弦相似度
            similarities = F.cosine_similarity(base_frame_tokens, mean_token, dim=1)
            
            # 选择相似度最小的top_k_initial个token
            _, top_initial_indices = torch.topk(similarities, k=top_k_initial, largest=False)
            sorted_initial_indices, _ = torch.sort(top_initial_indices)
            
            # Random drop到目标数量
            if sorted_initial_indices.size(0) > num_desired:
                # 生成随机排列并选择前num_desired个
                rand_perm = torch.randperm(sorted_initial_indices.size(0))
                selected_indices = rand_perm[:num_desired]
                # 获取对应的索引并保持顺序
                selected_sorted = sorted_initial_indices[selected_indices]
                selected_sorted, _ = torch.sort(selected_sorted)
                keep_indices[frame] = selected_sorted.tolist()
            else:
                keep_indices[frame] = sorted_initial_indices.tolist()
        
        # 提取保留的token
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 处理非基准帧（保持原有逻辑不变）
    non_base = [i for i in range(num_frames) if i not in base_frames_index]
    for i, frame in enumerate(non_base):
        ratio = scales[i]
        top_k = int(round(ratio * frame_token_length))
        if top_k == 0:
            keep_indices[frame] = []
        else:
            frame_scores = L1_distance[frame].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame] = top_indices.tolist()
        
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 合并所有保留的token
    valid_tokens = [tokens for tokens in modified_image_feature if len(tokens) > 0]
    combined_tokens = torch.cat(valid_tokens, dim=0) if valid_tokens else torch.tensor([])
    return combined_tokens
def select_token_base_by_dart_10_random_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio):
    frame_token_length = 196
    num_frames = image_feature.shape[0] // frame_token_length
    keep_indices = [[] for _ in range(num_frames)]
    modified_image_feature = [[] for _ in range(num_frames)]

    # 处理基准帧
    for frame in base_frames_index:
        # 直接计算最终需要保留的数量
        top_k_final = int(round(base_frame_retention_ratio * frame_token_length))
        if top_k_final <= 0:
            keep_indices[frame] = []
        else:
            # 获取当前帧的所有token
            base_frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            
            # 随机选择10个token
            num_selected = 10
            random_indices = torch.randperm(frame_token_length)[:num_selected]
            selected_tokens = base_frame_tokens[random_indices]
            
            # 计算所有token与这10个的余弦相似度
            similarities = F.cosine_similarity(
                base_frame_tokens.unsqueeze(1),  # [196, 1, dim]
                selected_tokens.unsqueeze(0),    # [1, 10, dim]
                dim=-1
            )
            
            # 计算每个token的总相似度分数
            scores = similarities.sum(dim=1)  # 使用sum或mean根据需求调整
            
            # 将选中的10个token分数置零
            scores[random_indices] = 0
            
            # 选择分数最低的token
            _, selected_indices = torch.topk(scores, k=top_k_final, largest=False)
            selected_sorted, _ = torch.sort(selected_indices)
            keep_indices[frame] = selected_sorted.tolist()
        
        # 提取保留的token
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 处理非基准帧（保持原有逻辑不变）
    non_base = [i for i in range(num_frames) if i not in base_frames_index]
    for i, frame in enumerate(non_base):
        ratio = scales[i]
        top_k = int(round(ratio * frame_token_length))
        if top_k == 0:
            keep_indices[frame] = []
        else:
            frame_scores = L1_distance[frame].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame] = top_indices.tolist()
        
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 合并所有保留的token
    valid_tokens = [tokens for tokens in modified_image_feature if len(tokens) > 0]
    combined_tokens = torch.cat(valid_tokens, dim=0) if valid_tokens else torch.tensor([])
    return combined_tokens

def select_token_base_by_dart_uniform10_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio):
    frame_token_length = 196
    num_frames = image_feature.shape[0] // frame_token_length
    keep_indices = [[] for _ in range(num_frames)]
    modified_image_feature = [[] for _ in range(num_frames)]

    # 处理基准帧
    for frame in base_frames_index:
        # 生成空间均匀分布的10个基准索引（基于14x14网格）
        rows = [3, 10]  # 均匀分布的y坐标
        cols = [1, 4, 7, 10, 13]  # 均匀分布的x坐标
        base_indices = []
        for y in rows:
            for x in cols:
                idx = y * 14 + x  # 转换为1D索引
                base_indices.append(idx)
        base_indices = torch.tensor(base_indices, device=image_feature.device)
        
        # 获取当前帧的所有token
        base_frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
        
        # 计算每个token与10个基准token的相似度
        base_tokens = base_frame_tokens[base_indices]  # (10, dim)
        
        # 计算所有token与基准token的余弦相似度矩阵
        similarities = F.cosine_similarity(
            base_frame_tokens.unsqueeze(1),  # (196, 1, dim)
            base_tokens.unsqueeze(0),        # (1, 10, dim)
            dim=2
        )  # (196, 10)
        
        # 计算每个token的总相似度分数
        scores = similarities.sum(dim=1)  # (196,)
        
        # 强制将基准token的分数设为最小值
        scores[base_indices] = 0.0
        
        # 计算需要保留的token总数
        top_k_final = int(round(base_frame_retention_ratio * frame_token_length))
        
        # 确保至少保留基准token，并处理边界情况
        required_keep = len(base_indices)
        if top_k_final < required_keep:
            # 当需求保留数小于基准token时，仅保留基准token
            selected_indices = base_indices
        else:
            # 获取非基准token的mask
            all_indices = torch.arange(frame_token_length, device=image_feature.device)
            mask = torch.ones_like(all_indices, dtype=torch.bool)
            mask[base_indices] = False
            
            # 从非基准token中选取剩余的
            _, non_base_selected = torch.topk(scores[mask], k=top_k_final - required_keep, largest=False)
            selected_non_base = all_indices[mask][non_base_selected]
            
            # 合并索引并排序
            selected_indices = torch.cat([base_indices, selected_non_base])
        
        # 最终排序并保存
        sorted_indices, _ = torch.sort(selected_indices)
        keep_indices[frame] = sorted_indices.tolist()
        modified_image_feature[frame] = base_frame_tokens[sorted_indices]

    # 处理非基准帧（保持原有逻辑不变）
    non_base = [i for i in range(num_frames) if i not in base_frames_index]
    for i, frame in enumerate(non_base):
        ratio = scales[i]
        top_k = int(round(ratio * frame_token_length))
        if top_k == 0:
            keep_indices[frame] = []
        else:
            frame_scores = L1_distance[frame].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame] = top_indices.tolist()
        
        if keep_indices[frame]:
            frame_tokens = image_feature[frame*frame_token_length : (frame+1)*frame_token_length]
            modified_image_feature[frame] = frame_tokens[keep_indices[frame]]

    # 合并所有保留的token
    valid_tokens = [tokens for tokens in modified_image_feature if len(tokens) > 0]
    combined_tokens = torch.cat(valid_tokens, dim=0) if valid_tokens else torch.tensor([])
    return combined_tokens
def frame_wise_compression_adaptive_ratio4_threhold10(image_feature,base_frame_retention_ratio=0.4):
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=1.0)
    scales=generate_scales_whith_mean_frame_score(image_feature)
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_adaptive_ratio4_threhold8(image_feature,base_frame_retention_ratio=0.4):
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.8)
    scales=generate_scales_whith_mean_frame_score(image_feature)
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_adaptive_ratio4_threhold7(image_feature,base_frame_retention_ratio=0.4):
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.7)
    scales=generate_scales_whith_mean_frame_score(image_feature)
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_adaptive_ratio4_threshold6(image_feature,base_frame_retention_ratio=0.4):
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.6)
    scales=generate_scales_whith_mean_frame_score(image_feature)
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_adaptive_ratio5(image_feature,base_frame_retention_ratio=0.5):
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.9)
    scales=generate_scales_whith_mean_frame_score(image_feature)
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_ratio3_dart_uniform10(image_feature,base_frame_retention_ratio=0.3):
    num_frames=image_feature.shape[0]//196
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.9)
    scales = [0.3] * num_frames
    combined_tokens=select_token_base_by_dart_uniform10_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_ratio3_dart_random10(image_feature,base_frame_retention_ratio=0.3):
    num_frames=image_feature.shape[0]//196
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.9)
    scales = [0.3] * num_frames
    combined_tokens=select_token_base_by_dart_10_random_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def frame_wise_compression_ratio5(image_feature,base_frame_retention_ratio=0.5):
    num_frames=image_feature.shape[0]//196
    base_frames_index, L1_score,L1_distance=select_base_frames(image_feature,threshold=0.9)
    scales = [0.5] * num_frames
    combined_tokens=select_token_base_by_frame_mean_nonbase_by_L1(image_feature, L1_distance, base_frames_index, scales, base_frame_retention_ratio)
    return combined_tokens
def select_frames_whith_big_cosine_similarity_frame_mean(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.5
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
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
    selected_frames = frames[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames.view(-1, dim)
    
    return pruned_tensor

def select_frames_whith_small_cosine_similarity_frame_mean(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, dim)
    prune_ratio=0.5
    num_frame = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frame, 196, -1)
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
    selected_frames = frames[sorted_indices]  # 形状 (k_frames, 196, dim)
    
    # 调整形状为 (总保留token数, dim)
    pruned_tensor = selected_frames.view(-1, dim)
    
    return pruned_tensor
def select_topk_token_whith_L1_distance(image_feature,l1_distance,base_frames_index,scales):
    frame_token_length = 196

    # 对于每个非 base frame，根据对应的 retention_ratio 计算需要保留的 token 数量，并用 topk 选取
    num_frames = image_feature.shape[0]//196
    keep_indices = [[] for _ in range(num_frames)]  # 初始化空容器
    modified_image_feature=[[] for _ in range(num_frames)]
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



def select_token_whith_most_attention_score(image_features, attention_scores):
    """
    image_features: 原始特征矩阵，形状为 (总token数, 特征维度) = (3136, 896)
    attention_scores: 注意力分数矩阵，形状需与image_features一致 (3136,)
    num_frames: 视频帧数 (默认16)
    tokens_per_frame: 每帧token数 (默认196)
    prune_ratio: 剪枝比例 (默认0.5)
    """
    # 重塑为帧优先的维度 (帧数, token数, 特征维度)
    tokens_per_frame=196,
    prune_ratio=0.5
    num_frames=image_features.size(0)//196
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

        keep_indices_sorted, _ = torch.sort(keep_indices)

        pruned_frame = frame_features[keep_indices_sorted]
        pruned_list.append(pruned_frame)
    
    # 拼接所有帧的结果
    pruned_features = torch.cat(pruned_list, dim=0)  # PyTorch版本
    return pruned_features

def select_token_whith_small_similiarity_whith_frame_mean(input_tensor, prune_ratio):
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

def select_least_similar_frames_mean_and_video_mean(image_feature: torch.Tensor) -> torch.Tensor:
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
def select_most_similar_frames_mean_and_video_mean(image_feature: torch.Tensor) -> torch.Tensor:
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
    _, indices = torch.topk(cos_sim, k=k, largest=True)
    
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
def select_frames_by_big_cosine_similarity_token_and_video_mean_token(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, 896维)
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算整个视频的全局平均token
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)
    
    # 计算每个token与全局平均token的余弦相似度
    similarities = F.cosine_similarity(
        frames,
        video_mean_token.expand_as(frames),  # 将全局平均扩展为与frames相同的形状
        dim=2  # 在896维度计算相似度
    )
    
    # 计算每帧的总得分（所有token相似度之和）
    frame_scores = similarities.sum(dim=1)
    
    # 确定保留的帧数（至少保留1帧）
    k_frames = max(1, int(round(0.5 * total_frames)))
    
    # 获取得分最高的k_frames个帧的索引
    _, frame_indices = torch.topk(frame_scores, k=k_frames, largest=True)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(frame_indices, dim=0)
    
    # 收集保留的帧的所有token
    pruned_frames = frames[sorted_indices]
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.reshape(-1, dim)
    
    return pruned_tensor
def select_frames_by_small_cosine_similarity_token_and_video_mean_token(input_tensor):
    # 调整张量形状为 (num_frames, 196 tokens/帧, 896维)
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算整个视频的全局平均token
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)  # 形状 (1, 1, 896)
    
    # 计算每个token与全局平均token的余弦相似度
    similarities = F.cosine_similarity(
        frames,
        video_mean_token.expand_as(frames),  # 将全局平均扩展为与frames相同的形状
        dim=2  # 在896维度计算相似度
    )
    
    # 计算每帧的总得分（所有token相似度之和）
    frame_scores = similarities.sum(dim=1)
    
    # 确定保留的帧数（至少保留1帧）
    k_frames = max(1, int(round(0.5 * total_frames)))
    
    # 获取得分最高的k_frames个帧的索引
    _, frame_indices = torch.topk(frame_scores, k=k_frames, largest=False)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(frame_indices, dim=0)
    
    # 收集保留的帧的所有token
    pruned_frames = frames[sorted_indices]
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.reshape(-1, dim)
    
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



def two_stage_retention_one_forth(image_feature, prune_ratio=0.5):
    frames_after_drop=select_frames_by_small_cosine_similarity_token_and_video_mean_token(image_feature)
    compressed_token=select_token_whith_small_similiarity_whith_frame_mean(frames_after_drop, prune_ratio)

    return compressed_token

def select_token_with_spatial_big_similarity_pruneratio5(input_tensor, prune_ratio=0.5, grid_size=14):
    """
    优化版跨帧空间位置相似度剪枝
    参数：
        input_tensor: 输入张量 (总token数, dim)
        prune_ratio: 剪枝比例（0~1）
        grid_size: 特征图尺寸（默认14x14）
    """
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)  # (n_frames, k1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size  # (n_frames, k1)
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储 (grid_size, grid_size, dim)
    history_map = torch.zeros((grid_size, grid_size, dim), 
                            device=input_tensor.device,dtype=input_tensor.dtype)
    
    # 批量计算跨帧相似度
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧的token和坐标
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]  # (k1,)
        curr_cols = cols[t]  # (k1,)
        
        # 从历史图中获取对应位置的token
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算余弦相似度（自动处理零向量）
        sim = F.cosine_similarity(curr_tokens, prev_tokens, dim=1)  # (k1,)
        cross_scores[t] = sim
        
        # 更新历史图（仅更新当前帧保留的位置）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于跨帧相似度保留k2个token
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=True)
    final_indices = sorted_indices.gather(1, final_idx)
    
    # 收集最终token并展平
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    return pruned.view(-1, dim)

def select_token_with_spatial_small_similarity_pruneratio5(input_tensor, prune_ratio=0.5, grid_size=14):
    """
    优化版跨帧空间位置相似度剪枝
    参数：
        input_tensor: 输入张量 (总token数, dim)
        prune_ratio: 剪枝比例（0~1）
        grid_size: 特征图尺寸（默认14x14）
    """
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)  # (n_frames, k1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size  # (n_frames, k1)
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储 (grid_size, grid_size, dim)
    history_map = torch.zeros((grid_size, grid_size, dim), 
                            device=input_tensor.device,dtype=input_tensor.dtype)
    
    # 批量计算跨帧相似度
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧的token和坐标
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]  # (k1,)
        curr_cols = cols[t]  # (k1,)
        
        # 从历史图中获取对应位置的token
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算余弦相似度（自动处理零向量）
        sim = F.cosine_similarity(curr_tokens, prev_tokens, dim=1)  # (k1,)
        cross_scores[t] = sim
        
        # 更新历史图（仅更新当前帧保留的位置）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于跨帧相似度保留k2个token
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=False)
    final_indices = sorted_indices.gather(1, final_idx)
    
    # 收集最终token并展平
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    return pruned.view(-1, dim)


def select_token_with_spatial_big_similarity_optimized_pruneratio3(input_tensor, prune_ratio=0.3, grid_size=14):
    """
    优化版跨帧空间位置相似度剪枝
    参数：
        input_tensor: 输入张量 (总token数, dim)
        prune_ratio: 剪枝比例（0~1）
        grid_size: 特征图尺寸（默认14x14）
    """
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)  # (n_frames, k1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size  # (n_frames, k1)
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储 (grid_size, grid_size, dim)
    history_map = torch.zeros((grid_size, grid_size, dim), 
                            device=input_tensor.device,dtype=input_tensor.dtype)
    
    # 批量计算跨帧相似度
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧的token和坐标
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]  # (k1,)
        curr_cols = cols[t]  # (k1,)
        
        # 从历史图中获取对应位置的token
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算余弦相似度（自动处理零向量）
        sim = F.cosine_similarity(curr_tokens, prev_tokens, dim=1)  # (k1,)
        cross_scores[t] = sim
        
        # 更新历史图（仅更新当前帧保留的位置）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于跨帧相似度保留k2个token
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=True)
    final_indices = sorted_indices.gather(1, final_idx)
    
    # 收集最终token并展平
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    return pruned.view(-1, dim)


def select_token_with_spatial_small_similarity_pruneratio3(input_tensor, prune_ratio=0.3, grid_size=14):
    """
    优化版跨帧空间位置相似度剪枝
    参数：
        input_tensor: 输入张量 (总token数, dim)
        prune_ratio: 剪枝比例（0~1）
        grid_size: 特征图尺寸（默认14x14）
    """
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)  # (n_frames, k1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size  # (n_frames, k1)
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储 (grid_size, grid_size, dim)
    history_map = torch.zeros((grid_size, grid_size, dim), 
                            device=input_tensor.device,dtype=input_tensor.dtype)
    
    # 批量计算跨帧相似度
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧的token和坐标
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]  # (k1,)
        curr_cols = cols[t]  # (k1,)
        
        # 从历史图中获取对应位置的token
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算余弦相似度（自动处理零向量）
        sim = F.cosine_similarity(curr_tokens, prev_tokens, dim=1)  # (k1,)
        cross_scores[t] = sim
        
        # 更新历史图（仅更新当前帧保留的位置）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于跨帧相似度保留k2个token
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=False)
    final_indices = sorted_indices.gather(1, final_idx)
    
    # 收集最终token并展平
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    return pruned.view(-1, dim)


def select_token_with_most_similar_L1_pruneratio3(input_tensor, prune_ratio=0.7, grid_size=14):
    """跨帧空间位置剪枝（使用L1距离）"""
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储（继承输入数据类型）
    history_map = torch.zeros((grid_size, grid_size, dim),
                            device=input_tensor.device,
                            dtype=input_tensor.dtype)
    
    # 初始化跨帧得分矩阵
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧数据
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]
        curr_cols = cols[t]
        
        # 获取历史token（自动处理未更新位置）
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算L1距离（绝对值之和）
        l1_dist = torch.abs(curr_tokens - prev_tokens).sum(dim=1)
        cross_scores[t] = l1_dist
        
        # 更新历史图（保留当前token）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于L1距离保留k2个token（选择距离最小的）
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=False)  # 改为False
    
    # 收集最终token
    final_indices = sorted_indices.gather(1, final_idx)
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    
    return pruned.view(-1, dim)


def select_token_with_least_similar_L1_pruneratio3(input_tensor, prune_ratio=0.7, grid_size=14):
    """跨帧空间位置剪枝（使用L1距离）"""
    # 转换为帧序列 (num_frames, tokens_per_frame, dim)
    num_frames = input_tensor.size(0) // (grid_size**2)
    frames = input_tensor.view(num_frames, grid_size**2, -1)
    n_frames, n_tokens, dim = frames.shape
    
    # 第一步：每帧保留k1个低相似度token
    k1 = int((1 - prune_ratio) * n_tokens)
    frame_means = frames.mean(dim=1, keepdim=True)
    similarities = F.cosine_similarity(frames, frame_means, dim=2)
    _, indices = torch.topk(similarities, k=k1, dim=1, largest=False)
    sorted_indices, _ = torch.sort(indices, dim=1)

    # 转换为二维坐标 (row, col)
    rows = sorted_indices // grid_size
    cols = sorted_indices % grid_size
    
    # 初始化历史token存储（继承输入数据类型）
    history_map = torch.zeros((grid_size, grid_size, dim),
                            device=input_tensor.device,
                            dtype=input_tensor.dtype)
    
    # 初始化跨帧得分矩阵
    cross_scores = torch.zeros((n_frames, k1), device=input_tensor.device)
    
    for t in range(num_frames):
        # 当前帧数据
        curr_tokens = frames[t, sorted_indices[t]]  # (k1, dim)
        curr_rows = rows[t]
        curr_cols = cols[t]
        
        # 获取历史token（自动处理未更新位置）
        prev_tokens = history_map[curr_rows, curr_cols]  # (k1, dim)
        
        # 计算L1距离（绝对值之和）
        l1_dist = torch.abs(curr_tokens - prev_tokens).sum(dim=1)
        cross_scores[t] = l1_dist
        
        # 更新历史图（保留当前token）
        history_map[curr_rows, curr_cols] = curr_tokens.detach().to(history_map.dtype)

    # 第二步：基于L1距离保留k2个token（选择距离最小的）
    k2 = int((1 - prune_ratio) * n_tokens)
    _, final_idx = torch.topk(cross_scores, k=k2, dim=1, largest=True)  # 改为False
    
    # 收集最终token
    final_indices = sorted_indices.gather(1, final_idx)
    pruned = frames.gather(1, final_indices.unsqueeze(-1).expand(-1, -1, dim))
    
    return pruned.view(-1, dim)