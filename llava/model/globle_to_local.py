import torch
import random
import torch.nn.functional as F
import numpy as np

def select_base_frames_(image_feature,threshold=0.8):
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
        print("l1_score",l1_score)
        if l1_score > threshold:

            base_frames_index.append(i)       # 记录新 clip 的 base frame 索引
            current_base = frame2_tokens
            L1_score.append(l1_score)
            L1_distance.append(l1_distance)
        else:
            # 若距离未超过阈值，则当前帧属于当前 clip
            L1_score.append(l1_score)
            L1_distance.append(l1_distance)    
    
    return base_frames_index,L1_distance


def select_base_frames_by_neighbor_l1(video_tensor, threshold=0.75):
    """
    video_tensor: shape [total_tokens, channels]
    threshold: L1差异阈值
    token_per_frame: 每帧包含的token数量
    返回值: 包含关键帧索引的列表（始终包含第0帧）
    """
    # 计算总帧数
    token_per_frame=196
    frame_num = video_tensor.size(0) // token_per_frame
    
    # 重塑张量维度为 [帧数, 每帧token数, 通道数]
    frames = video_tensor.view(frame_num, token_per_frame, -1)
    
    # 初始化结果列表（自动包含第0帧）
    base_frames = [0]
    l1_distance_list=[]
    l1_distance_list.append(torch.full((196,), 0.0, device=video_tensor.device))

    
    # 遍历后续帧计算差异
    for i in range(1, frame_num):
        # 计算当前帧与前一帧的L1距离
        l1_distance = torch.abs(frames[i] - frames[i-1]).mean(dim=-1)
        l1_diff=l1_distance.mean()

        
        # 若超过阈值则记录当前帧索引
        if l1_diff > threshold:
            base_frames.append(i)
            l1_distance_list.append(torch.full((196,), 0.0, device=video_tensor.device))

        else:
            l1_distance_list.append(l1_distance)

    combined_tensor = torch.cat(l1_distance_list)
    return base_frames,combined_tensor

def generate_clip_scales(base_frames_index, token_score, base_scale=0.5, temperature=0.1):
    # 确保输入为张量
    device = token_score.device 
    token_score = torch.tensor(token_score).to(token_score.device) if not isinstance(token_score, torch.Tensor) else token_score
    token_score=token_score.float() 
    base_frames = torch.tensor(sorted(base_frames_index), device=device)  # 关键修改：指定设备
    
    # 创建clip的mask矩阵
    frame_number = token_score.size(0)
    clip_mask = torch.zeros(len(base_frames), frame_number, device=device)  # 指定设备
    clip_mask.scatter_(1, base_frames.unsqueeze(1), 1)
    clip_mask = clip_mask.cumsum(dim=0).clamp(max=1)
    clip_mask = clip_mask.type(token_score.dtype)
    
    # 计算每个clip的总分（均值）
    clip_scores = (clip_mask @ token_score) / clip_mask.sum(dim=1)
    
    # 使用向量化操作计算softmax
    shifted_scores = (clip_scores - clip_scores.max()) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (exp_scores.sum() + 1e-8)
    
    # 计算每个clip的缩放比例
    scales = base_scale * (1 + softmax_scores - softmax_scores.mean())
    scales = torch.clamp(scales, max=1.0)  # 确保不超过1.0

    return scales
def generate_scales_whith_score_(base_frames_index, score, clip_scales, temperature=0.1):
    token_per_frame = 196
    total_token = score.shape[0]
    frame_number = total_token // token_per_frame
    clip_number = len(base_frames_index)
    frame_score = score.view(frame_number, token_per_frame).sum(-1)

    final_scales = []
    for i in range(clip_number):
        if i == clip_number - 1:
            current_clip_score = frame_score[base_frames_index[i]:]
        else:
            current_clip_score = frame_score[base_frames_index[i]:base_frames_index[i + 1]]

        current_clip_length = current_clip_score.shape[0]
        current_clip_scale = clip_scales[i]

        # 显式处理单帧clip的情况
        if current_clip_length == 1:
            # 将标量转换为形状为(1,)的Tensor
            scales = current_clip_scale.unsqueeze(0)  # 或 torch.tensor([current_clip_scale], device=current_clip_scale.device)
        else:
            shifted_scores = (current_clip_score - torch.max(current_clip_score)) / temperature
            exp_scores = torch.exp(shifted_scores)
            softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)

            if torch.sum(softmax_scores) == 0:
                # 创建全为current_clip_scale的Tensor
                scales = torch.full((current_clip_length,), current_clip_scale, device=current_clip_scale.device)
            else:
                scales = current_clip_scale * (1 + softmax_scores - torch.mean(softmax_scores))
                scales = torch.clip(scales, None, 1.0)

        final_scales.append(scales)

    return torch.cat(final_scales)
def generate_frame_mean_score(input_tensor):

    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧需要保留的token数量（四舍五入）
    
    # 计算每帧的平均token
    avg_per_frame = frames.mean(dim=1, keepdim=True)  # 形状 (num_frames, 1, 896)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, avg_per_frame.expand_as(frames), dim=2)

    frame_mean_score=similarities.flatten() 

    return frame_mean_score

def generate_clip_mean_score(input_tensor, base_frames_index):
    """
    将每个clip内的token mean成一个token，然后计算clip内所有token和这个token的cosine similarity。
    
    参数:
        input_tensor (torch.Tensor): 输入张量，形状为 (total_tokens, dim)。
        base_frames_index (list): 每个clip的起始帧索引。
    
    返回:
        List[torch.Tensor]: 每个clip内token与mean token的余弦相似度列表。
    """
    frame_token_length = 196  # 每帧的token数量
    num_frames = input_tensor.shape[0] // frame_token_length  # 总帧数
    clips = []  # 存储每个clip的起始和结束索引
    
    # 确定每个clip的起始和结束索引
    for i in range(len(base_frames_index)):
        start = base_frames_index[i]
        if i < len(base_frames_index) - 1:
            end = base_frames_index[i + 1] - 1
        else:
            end = num_frames - 1
        clips.append((start, end))
    
    clip_similarities = []  # 存储每个clip的相似度结果
    
    for start, end in clips:
        # 获取当前clip的所有token
        clip_start_idx = start * frame_token_length
        clip_end_idx = (end + 1) * frame_token_length
        clip_tokens = input_tensor[clip_start_idx:clip_end_idx]
        
        # 计算clip的mean token
        mean_token = clip_tokens.mean(dim=0, keepdim=True)  # 形状 (1, dim)
        
        # 计算每个token与mean token的余弦相似度
        similarities = F.cosine_similarity(clip_tokens, mean_token.expand_as(clip_tokens), dim=1)
        
        # 保存当前clip的相似度结果
        clip_similarities.append(similarities)
    clip_mean_score = torch.cat(clip_similarities, dim=0)

    
    return clip_mean_score


def generate_video_mean_score(input_tensor):


    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    total_frames, tokens_per_frame, dim = frames.shape
    
    # 计算每帧需要保留的token数量（四舍五入）
    
    # 计算每帧的平均token
    video_mean_token = frames.mean(dim=(0, 1), keepdim=True)
    
    # 计算每个token与平均token的余弦相似度
    similarities = F.cosine_similarity(frames, video_mean_token.expand_as(frames), dim=2)
    video_mean_score=similarities.flatten()

    return video_mean_score




# def select_token(image_feature, token_score, scales):
#     frame_token_length = 196
#     num_frames = image_feature.shape[0] // frame_token_length
#     frame = image_feature.view(num_frames, frame_token_length, -1)
#     frame_score = token_score.view(num_frames, frame_token_length)
#     kept_frames = []
#     frame_token_indices = []    

#     for i in range(num_frames):
#         ratio = scales[i].item()  
#         r_token_number = int(ratio * frame_token_length)
        
#         _, top_indices = torch.topk(frame_score[i], r_token_number, largest=False, sorted=False)
#         sorted_indices, _ = torch.sort(top_indices, dim=0)
        
#         sorted_indices = sorted_indices.to(frame[i].device)
#         frame_token_indices.append(sorted_indices)
#         pruned_frame = frame[i].gather(
#             dim=1,
#             index=sorted_indices.unsqueeze(1).expand(-1, frame[i].shape[-1])
#         )
#         kept_frames.append(pruned_frame)
#     return torch.cat(kept_frames),frame_token_indices

def prune_video_tensor(video_tensor, token_score, scales):
    frame_token_length = 196
    num_frames = video_tensor.shape[0] // frame_token_length
    frame = video_tensor.view(num_frames, frame_token_length, -1)
    frame_score = token_score.view(num_frames, frame_token_length)
    kept_frames = []

    for i in range(num_frames):
        ratio = scales[i].item()  # 获取当前帧的保留比例
        r_token_number = int(ratio * frame_token_length)  # 计算保留的token数量

        # 获取当前帧分数最低的r_token_number个token的索引
        _, top_indices = torch.topk(
            frame_score[i], 
            r_token_number, 
            largest=False,  # 选择最小值（低分代表需要保留的token）
            sorted=False
        )
        sorted_indices = torch.sort(top_indices)[0]  # 对索引排序

        # 按索引选取当前帧的token
        pruned_frame = frame[i].index_select(
            0, 
            sorted_indices
        )
        kept_frames.append(pruned_frame)

    # 将所有帧的token拼接成最终的pruned视频张量
    pruned_video = torch.cat(kept_frames, dim=0)
    return pruned_video
def generate_token_importance(input_tensor,base_frames_index):
    # clip_mean_score=generate_clip_mean_score(input_tensor, base_frames_index)
    frame_mean_score=generate_frame_mean_score(input_tensor)
    # video_mean_score=generate_video_mean_score(input_tensor)
    # token_importance=clip_mean_score+frame_mean_score+video_mean_score

    return frame_mean_score


