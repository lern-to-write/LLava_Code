import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def generate_scale_for_frame(last_layer_attention, all_image_token_length, frame_number, frame_token_length, system_token_length,base_scale, temperature=10.0):
    # Calculate the scores for each token in the frame
    
    score_before_image=last_layer_attention.mean(dim=1)[0][system_token_length:system_token_length+all_image_token_length, :system_token_length]
    score_after_image=last_layer_attention.mean(dim=1)[0][system_token_length:system_token_length+all_image_token_length,all_image_token_length+system_token_length:]
    unsequnse_score_in_globle=torch.cat((score_before_image, score_after_image), dim=-1).sum(dim=-1)
    score_in_globle=unsequnse_score_in_globle.view(frame_number,frame_token_length)
    frame_accumulate_score=score_in_globle.sum(dim=-1)
    # Normalize the scores and apply softmax
    shifted_scores = (frame_accumulate_score - torch.max(frame_accumulate_score)) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
    # Calculate scales ensuring no scale exceeds 1
    if torch.sum(softmax_scores) == 0:
        scales = [base_scale] * frame_number
    else:
        scales = base_scale * (1 + softmax_scores - torch.mean(softmax_scores))
        scales = torch.clip(scales, None, 1.0)

    return scales
def generate_scale_for_frame_with_system_token(last_layer_attention, all_image_token_length, frame_number, frame_token_length,system_token_length,user_instruction_length,base_scale, temperature=10.0):
    # Calculate the scores for each token in the frame

    before_score_before_image=last_layer_attention.mean(dim=1)[0][system_token_length+all_image_token_length:system_token_length+all_image_token_length+user_instruction_length]
    score_before_image=before_score_before_image[:,system_token_length:system_token_length+all_image_token_length].transpose(0,1).sum(-1)

    score_in_globle=score_before_image.view(frame_number,frame_token_length)
    frame_accumulate_score=score_in_globle.sum(dim=-1)
    # Normalize the scores and apply softmax
    shifted_scores = (frame_accumulate_score - torch.max(frame_accumulate_score)) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
    # Calculate scales ensuring no scale exceeds 1
    if torch.sum(softmax_scores) == 0:
        scales = [base_scale] * frame_number
    else:
        scales = base_scale * (1 + softmax_scores - torch.mean(softmax_scores))
        scales = torch.clip(scales, None, 1.0)

    return scales

def fastv_score(layer_outputs, all_image_token_length, system_token_length):
    last_layer_attention = layer_outputs[1]
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    last_layer_attention_score = last_layer_attention_avg[-1][system_token_length:system_token_length+all_image_token_length]
    return last_layer_attention_score
def select_topk_token_index_each_frame(hidden_states, score_in_globle, all_image_token_length, score_in_local_frame,frame_number,system_token_length, frame_token_length, scales,device):

    #norm_score_in_globle = F.normalize(score_in_globle.unsqueeze(0), p=2, dim=-1).squeeze(0)
    #norm_score_in_local_frame = F.normalize(score_in_local_frame.unsqueeze(0), p=2, dim=-1).squeeze(0)
    #combined_scores = norm_score_in_globle + norm_score_in_local_frame
    combined_scores = score_in_globle + score_in_local_frame
    seq_length = hidden_states.shape[1]
    assert len(scales) == frame_number, "topk_values 列表长度必须与 frame_number 一致"
    selected_indices = []
    for i in range(frame_number):
        start_idx = i * frame_token_length
        end_idx = start_idx + frame_token_length
        frame_tensor = combined_scores[start_idx:end_idx]
       
        topk_value = int(frame_token_length * scales[i])
        topk_indices = torch.topk(frame_tensor, topk_value).indices
        topk_indices = topk_indices + start_idx  # 转换为全局索引
        selected_indices.append(topk_indices)
    
    selected_indices=torch.cat(selected_indices)+system_token_length+1
    keep_indexs = torch.cat((torch.arange(system_token_length,device=device), selected_indices, torch.arange(system_token_length+all_image_token_length,seq_length,device=device)))
    keep_indexs = selected_indices.sort().values
    return keep_indexs
def select_topk_token_index_each_frame_only_whith_one_score(hidden_states, all_image_token_length, score_in_local_frame,frame_number,system_token_length, frame_token_length, scales,device):
    score_reshaped = score_in_local_frame.view(frame_number, frame_token_length)
    num_tokens_to_keep = (scales * frame_token_length).int()
    indices_to_keep = []
    image_indices_to_keep=[]
    for i in range(16):
        frame_score = score_reshaped[i]
        num_keep = num_tokens_to_keep[i]
        _, top_indices = torch.topk(frame_score, num_keep)
        original_indices = top_indices + i * frame_token_length
        indices_to_keep.extend(original_indices.tolist())
        image_indices_to_keep.append(top_indices.tolist())
    indices_to_keep = torch.tensor(indices_to_keep).to(device)
    top_attention_rank_index=indices_to_keep+system_token_length
    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), top_attention_rank_index, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs,image_indices_to_keep
#在一个frame中该token与其他token的注意力分数之和
def get_score_in_local_frame(last_layer_attention, all_image_token_length, frame_token_length, frame_number, system_token_length):
    attention_matrix = last_layer_attention.mean(dim=1)[0]
    image_attention_matrix=attention_matrix[system_token_length:system_token_length+all_image_token_length,system_token_length:system_token_length+all_image_token_length]
    local_attention_sums=[]

    for frame_idx in range(frame_number):
        start_idx = frame_idx * frame_token_length
        end_idx = start_idx + frame_token_length
        frame_attention = image_attention_matrix[start_idx:end_idx, start_idx:end_idx]
        frame_attention_sums = frame_attention.sum(dim=1)
        local_attention_sums.append(frame_attention_sums)

    local_attention_sums = torch.cat(local_attention_sums)
    
    return local_attention_sums
#该token和所有非图像token注意力分数之和
def get_score_in_globle(last_layer_attention, all_image_token_length, frame_number, frame_token_length, system_token_length): 
    
    score_before_image=last_layer_attention.mean(dim=1)[0][system_token_length:system_token_length+all_image_token_length, :system_token_length]
    score_after_image=last_layer_attention.mean(dim=1)[0][system_token_length:system_token_length+all_image_token_length,all_image_token_length+system_token_length:]
    score_in_globle=torch.cat((score_before_image, score_after_image), dim=-1).sum(dim=-1)
    
    return score_in_globle


def generate_scale_for_frame_with_frame_attn(last_layer_attention, all_image_token_length, frame_number, frame_token_length,system_token_length,user_instruction_length,base_scale, temperature=10.0):
    # Calculate the scores for each token in the frame

    image_attention_matrix=last_layer_attention.mean(dim=1)[0]
    frame_score_list=[]
    for frame_idx in range(frame_number):
        start_idx = frame_idx * frame_token_length
        end_idx = start_idx + frame_token_length
        frame_attention = image_attention_matrix[start_idx:end_idx, system_token_length:system_token_length+all_image_token_length]
        frame_attention_sums = frame_attention.sum()
        frame_score_list.append(frame_attention_sums)

    frame_accumulate_score=torch.stack(frame_score_list)
    # Normalize the scores and apply softmax
    shifted_scores = (frame_accumulate_score - torch.max(frame_accumulate_score)) / temperature
    exp_scores = torch.exp(shifted_scores)
    softmax_scores = exp_scores / (torch.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero
    # Calculate scales ensuring no scale exceeds 1
    if torch.sum(softmax_scores) == 0:
        scales = [base_scale] * frame_number
    else:
        scales = base_scale * (1 + softmax_scores - torch.mean(softmax_scores))
        scales = torch.clip(scales, None, 1.0)
    return scales



def select_token_big_max_hs_protected(hidden_states,system_token_length,all_image_token_length,protected_indices,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    sim_matrix = F.cosine_similarity(
        all_image_token.unsqueeze(1),
        all_instruction_token.unsqueeze(0),
        dim=2
    )
    scores, _ = torch.max(sim_matrix, dim=1) 

    # 新增：强制保留protected_indices中的token
    if protected_indices is not None:
        # 将全局索引转换为图像区域内的相对索引（假设protected_indices是全局索引）
        image_protected = protected_indices
        valid_mask = (image_protected >= 0) & (image_protected < all_image_token_length)
        image_protected = image_protected[valid_mask]
        
        if image_protected.numel() > 0:
            # 将保护token的评分设为无穷大，确保被选中
            scores[image_protected] = float('inf')


    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=True)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs

def select_token_big_max_hs(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    # sim_matrix = F.cosine_similarity(
    #     all_image_token.unsqueeze(1),
    #     all_instruction_token.unsqueeze(0),
    #     dim=2
    # )
    # scores, _ = torch.max(sim_matrix, dim=1) 


        # 余弦相似度公式：sim(a,b) = (a·b) / (|a|·|b|)
    dot_product = all_image_token @ all_instruction_token.T  # [image_tokens, instruction_tokens]
    norm_image = all_image_token.norm(p=2, dim=1, keepdim=True)  # [image_tokens, 1]
    norm_instr = all_instruction_token.norm(p=2, dim=1)  # [instruction_tokens]

    # 分母矩阵（避免显存爆炸）
    denominator = norm_image @ norm_instr.unsqueeze(0)  # [image_tokens, instruction_tokens]

    sim_matrix = dot_product / (denominator + 1e-8)  # 防止除零

    # 直接取每行最大值
    scores, _ = torch.max(sim_matrix, dim=1)

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=False)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs


def select_token_big_max_hs_ratio_0_point_25(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    sim_matrix = F.cosine_similarity(
        all_image_token.unsqueeze(1),
        all_instruction_token.unsqueeze(0),
        dim=2
    )
    scores, _ = torch.max(sim_matrix, dim=1) 

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=True)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs

def select_token_big_max_hs_ratio_0_point_75(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.75) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    sim_matrix = F.cosine_similarity(
        all_image_token.unsqueeze(1),
        all_instruction_token.unsqueeze(0),
        dim=2
    )
    scores, _ = torch.max(sim_matrix, dim=1) 

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=True)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs

def select_token_small_max_hs(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    sim_matrix = F.cosine_similarity(
        all_image_token.unsqueeze(1),
        all_instruction_token.unsqueeze(0),
        dim=2
    )
    scores, _ = torch.max(sim_matrix, dim=1) 

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=False)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs



def select_token_big_mean_hs(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    sim_matrix = F.cosine_similarity(
        all_image_token.unsqueeze(1),
        all_instruction_token.unsqueeze(0),
        dim=2
    )
    scores = torch.mean(sim_matrix, dim=1) 

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token

    # 筛选topk tokens
    _, topk_indices = torch.topk(scores, k, largest=True)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs




def select_token_small_mean_hs(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    # sim_matrix = F.cosine_similarity(
    #     all_image_token.unsqueeze(1),
    #     all_instruction_token.unsqueeze(0),
    #     dim=2
    # )
    # scores = torch.mean(sim_matrix, dim=1) 
   # 手动计算余弦相似度
    dot_product = torch.matmul(
        all_image_token,  # [num_image_tokens, hidden_dim]
        all_instruction_token.T  # [hidden_dim, num_instruction_tokens]
    )  # 结果形状为 [num_image_tokens, num_instruction_tokens]

    # 计算图像token和指令token的L2范数
    image_norms = torch.norm(all_image_token, p=2, dim=1, keepdim=True)  # [num_image_tokens, 1]
    instruction_norms = torch.norm(all_instruction_token, p=2, dim=1, keepdim=True).T  # [1, num_instruction_tokens]

    # 避免除零错误，增加一个小的常数
    denominator = image_norms @ instruction_norms + 1e-8  # [num_image_tokens, num_instruction_tokens]

    # 计算余弦相似度矩阵
    sim_matrix = dot_product / denominator

    # 对每一行取平均值作为分数
    scores = torch.mean(sim_matrix, dim=1)
    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token


    _, topk_indices = torch.topk(scores, k, largest=False)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs



def select_token_big_mean_hs(hidden_states,system_token_length,all_image_token_length,keep_ratio= 0.5) -> torch.Tensor:
    device=hidden_states.device
    all_image_token=hidden_states[0][system_token_length:system_token_length+all_image_token_length]
    all_instruction_token=hidden_states[0][system_token_length+all_image_token_length:]

    # sim_matrix = F.cosine_similarity(
    #     all_image_token.unsqueeze(1),
    #     all_instruction_token.unsqueeze(0),
    #     dim=2
    # )
    # scores = torch.mean(sim_matrix, dim=1) 
   # 手动计算余弦相似度
    dot_product = torch.matmul(
        all_image_token,  # [num_image_tokens, hidden_dim]
        all_instruction_token.T  # [hidden_dim, num_instruction_tokens]
    )  # 结果形状为 [num_image_tokens, num_instruction_tokens]

    # 计算图像token和指令token的L2范数
    image_norms = torch.norm(all_image_token, p=2, dim=1, keepdim=True)  # [num_image_tokens, 1]
    instruction_norms = torch.norm(all_instruction_token, p=2, dim=1, keepdim=True).T  # [1, num_instruction_tokens]

    # 避免除零错误，增加一个小的常数
    denominator = image_norms @ instruction_norms + 1e-8  # [num_image_tokens, num_instruction_tokens]

    # 计算余弦相似度矩阵
    sim_matrix = dot_product / denominator

    # 对每一行取平均值作为分数
    scores = torch.mean(sim_matrix, dim=1)
    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留1个token


    _, topk_indices = torch.topk(scores, k, largest=True)
    topk_indices = topk_indices.sort()[0]
    o_indice=topk_indices+system_token_length

    keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), o_indice, torch.arange(system_token_length+all_image_token_length,hidden_states.shape[1],device=device)))
    return keep_indexs


def select_token_with_mean_instruction(hidden_states, system_token_length, all_image_token_length,user_instruction_length,keep_ratio=0.5) -> torch.Tensor:
    device = hidden_states.device
    
    # 提取图像 token 和指令 token
    all_image_token = hidden_states[0][system_token_length:system_token_length + all_image_token_length]

    all_instruction_token = hidden_states[0][system_token_length + all_image_token_length:]
    
    # 将指令 token 压缩为一个 token（取 mean）
    mean_all_instruction_token = torch.mean(all_instruction_token, dim=0, keepdim=True)  # [1, hidden_dim]
    
    # 手动计算余弦相似度
    dot_product = torch.matmul(
        all_image_token,  # [num_image_tokens, hidden_dim]
        mean_all_instruction_token.T  # [hidden_dim, 1]
    )  # 结果形状为 [num_image_tokens, 1]

    # 计算图像 token 和指令 token 的 L2 范数
    image_norms = torch.norm(all_image_token, p=2, dim=1, keepdim=True)  # [num_image_tokens, 1]
    instruction_norm = torch.norm(mean_all_instruction_token, p=2, dim=1, keepdim=True).T  # [1, 1]

    # 避免除零错误，增加一个小的常数
    denominator = image_norms @ instruction_norm + 1e-8  # [num_image_tokens, 1]

    # 计算余弦相似度
    cosine_sim = dot_product / denominator  # [num_image_tokens, 1]
    scores = cosine_sim.squeeze(-1)  # [num_image_tokens]

    # 计算保留数量
    num_tokens = all_image_token.size(0)
    k = int(num_tokens * keep_ratio)
    k = max(k, 1)  # 至少保留 1 个 token

    # 筛选 top-k tokens
    _, topk_indices = torch.topk(scores, k, largest=False)
    topk_indices = topk_indices.sort()[0]  # 排序以确保顺序
    o_indice = topk_indices + system_token_length
    

    # 构造最终保留的索引
    keep_indexs = torch.cat((
        torch.arange(system_token_length, device=device),  # 系统 token
        o_indice,  # 图像 token 中保留的部分
        torch.arange( hidden_states.shape[1]-user_instruction_length , hidden_states.shape[1], device=device)  # 其他部分
    ))
    return keep_indexs