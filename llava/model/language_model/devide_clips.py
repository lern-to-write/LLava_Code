import torch
import numpy as np
import math
def split_video(hidden_states,system_token_length, all_image_token_length,frame_number, frame_token_length,threshold=0.6):
    # 计算相邻帧的L1距离
    video_tensor=hidden_states[0][system_token_length:system_token_length+all_image_token_length].view(frame_number,frame_token_length,-1)
    dists = []
    
    for i in range(video_tensor.size(0) - 1):
        frame_i = video_tensor[i]
        frame_j = video_tensor[i+1]
        l1_dist = torch.abs(frame_i - frame_j).mean(dim=-1).mean(dim=-1)
        # normed_frame_i=torch.norm(frame_i)
        # normed_frame_j=torch.norm(frame_j)
        # l1_dist=torch.dist(frame_i,frame_j,p=2)
        # l1_dist=torch.cdist(normed_frame_i,normed_frame_j,p=1)
        dists.append(l1_dist)
    split_indices = [i+1 for i, d in enumerate(dists) if d > threshold]
    split_points = [0] + split_indices + [video_tensor.size(0)]
    # 切割clips
    clips = []
    for k in range(len(split_points)-1):
        start = split_points[k]
        end = split_points[k+1]
        clips.append(video_tensor[start:end])
    # 切割对应的distance列表
    distance_clips = []
    for k in range(len(split_points)-1):
        start = split_points[k]
        end_clip = split_points[k+1]
        start_dist = start
        end_dist = end_clip - 1  # 转换为distance切片结束位置
        distance_clip = dists[start_dist:end_dist]
        distance_clips.append(distance_clip)
    return clips, distance_clips

def allocate_retention_rates(distance_clips, clips,all_image_token_length,base_ratio=0.25):
    # assert len(distance_clips) == len(clips), "distance_clips与clips长度不一致"
    num_clips = len(clips)
    if num_clips == 0:
        return []
    retention_clips = []
    total_retention_token_number=all_image_token_length*base_ratio
    token_retention_token_number_for_clip=total_retention_token_number/num_clips
    for clip_idx in range(num_clips):
        # 获取当前clip信息
        # n_frames = clips[clip_idx].shape[0]
        # distances = torch.tensor(distance_clips[clip_idx])
        clip_frame_number=clips[clip_idx].shape[0]
        clip_frame_token_number=clips[clip_idx].shape[1]
        token_ratention_ratio_for_clips=token_retention_token_number_for_clip/(clip_frame_number*clip_frame_token_number)
        # # Normalize the scores and apply softmax
        # shifted_scores = (distances - torch.max(distances)) / 10.0
        # exp_scores = torch.exp(shifted_scores)
        # softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
        # # Calculate scales ensuring no scale exceeds 1
        
        # # Calculate scales ensuring no scale exceeds 1
        # if torch.sum(softmax_scores) == 0:
        #     scales = [base_ratio] * n_frames
        # else:
        #     scales = base_ratio * (1 + softmax_scores - torch.mean(softmax_scores))
        #     scales = torch.clip(scales, None, 1.0)
        scales = torch.full(size=(clips[clip_idx].shape[0],), fill_value=token_ratention_ratio_for_clips)
        retention_clips.append(scales)
    retention_ratio_frames= torch.cat(retention_clips,dim=0)
    
    return retention_ratio_frames


def get_score_whith_L1(video_tensor,threshold):
    base_frames_index = [0]           # 保存各 clip 的 base frame 索引
    clip_index_list = []                  # 保存每个 clip 内所有帧的索引
    current_clip = [0]          # 当前 clip 的帧索引列表，初始帧为 0
    current_base = video_tensor[0]
    L1_score=[]
    L1_distance=[]
    # 从第 1 帧开始遍历视频
    for i in range(1, video_tensor.size(0)):
        # 计算当前帧与当前 base frame 之间的 L1 距离
        l1_distance = torch.abs(video_tensor[i] - current_base)
        l1_score = l1_distance.mean().item()
        if l1_score > threshold:
            # 当 L1 距离大于阈值，认为当前帧为新 clip 的开始
            base_frames_index.append(i)       # 记录新 clip 的 base frame 索引
            #clip_index_list.append(current_clip)    # 保存上一个 clip 的所有帧索引
            #current_clip = [i]          # 初始化新 clip，从当前帧开始
            current_base = video_tensor[i]  # 更新 base frame
        else:
            # 若距离未超过阈值，则当前帧属于当前 clip
            current_clip.append(i)
            L1_score.append(l1_score)
            L1_distance.append(l1_distance)
    
    # 添加最后一个 clip
    clip_index_list.append(current_clip)
    return base_frames_index, clip_index_list, L1_score,L1_distance



def select_topk_token_whith_L1_distance(video_tensor,l1_distance,base_frames_index,scales):
    token_num = video_tensor.size(1)
    indices_list = []
    # 对于每个非 base frame，根据对应的 retention_ratio 计算需要保留的 token 数量，并用 topk 选取
    num_frames = video_tensor.size(0)
    keep_indices = [[] for _ in range(num_frames)]  # 初始化空容器
    # 处理基准帧（保留全部token）
    for frame in base_frames_index:
        keep_indices[frame] = np.arange(token_num)
    # 处理非基准帧
    non_base = [i for i in range(num_frames) if i not in base_frames_index]
    for i, frame in enumerate(non_base):
        ratio = scales[frame]
        top_k = int(round(ratio * token_num))
        if top_k == 0:  # 不保留任何token
            keep_indices[frame] = []
        else:  # 按得分选择top_k
            frame_scores = l1_distance[i].sum(-1)
            _, top_indices = torch.topk(frame_scores, top_k)
            keep_indices[frame]=top_indices

    # 将结果转换为numpy数组列表
    return keep_indices


def generate_scales_whith_score(base_frames_index,L1_score,frame_number,base_scale,temperature):
    # Normalize the scores and apply softmax
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
    
def convert_index(index, input_size=27, output_size=14, padding=1, stride=2):
    """
    将原始一维索引转换为经过2D池化后的一维索引。
    
    参数：
      index: 原始索引（0到728之间）
      input_size: 原始frame的宽度/高度（假设为正方形，这里是27）
      output_size: 池化后frame的宽度/高度（这里是14）
      padding: 池化时的padding（这里为1）
      stride: 池化的步长（这里为2）
    
    返回：
      池化后对应的一维索引
    """
    # 将一维索引转换为二维坐标
    r = index // input_size
    c = index % input_size

    # 考虑padding后的坐标调整
    r_adj = r + padding
    c_adj = c + padding

    # 计算池化后坐标（注意这里采用整除）
    r_p = r_adj // stride
    c_p = c_adj // stride

    # 将池化后二维坐标转换为一维索引
    new_index = r_p * output_size + c_p
    return new_index

    


