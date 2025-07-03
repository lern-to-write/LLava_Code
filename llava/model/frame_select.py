import torch
import random
import torch.nn.functional as F
def select_frame_by_keep_three(feature_tensor):
    """
    每四帧保留前三帧，去掉最后一帧，从第0帧开始
    """
    tokens_per_frame = 196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame

    # Reshape the tensor to (n_frames, tokens_per_frame, -1)
    frames = feature_tensor.view(n_frames, tokens_per_frame, -1)

    # Create a list to store indices of frames to keep
    keep_indices = []

    # Iterate over frames in steps of 4
    for i in range(0, n_frames, 4):
        # Keep frames i, i+1, i+2
        keep_indices.extend(range(i, min(i + 3, n_frames)))

    # Select the frames to keep
    pruned_frames = frames[keep_indices]

    # Reshape the pruned frames back to the original shape
    pruned = pruned_frames.reshape(-1, feature_tensor.size(-1))
    return pruned
def select_frame_by_keep_two(feature_tensor):
    """
    每三帧保留前两帧，去掉最后一帧，从第0帧开始
    """
    tokens_per_frame = 196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame

    # Reshape the tensor to (n_frames, tokens_per_frame, -1)
    frames = feature_tensor.view(n_frames, tokens_per_frame, -1)

    # Create a list to store indices of frames to keep
    keep_indices = []

    # Iterate over frames in steps of 4
    for i in range(0, n_frames, 3):
        # Keep frames i, i+1, i+2
        keep_indices.extend(range(i, min(i + 2, n_frames)))

    # Select the frames to keep
    pruned_frames = frames[keep_indices]

    # Reshape the pruned frames back to the original shape
    pruned = pruned_frames.reshape(-1, feature_tensor.size(-1))

    return pruned
def select_frame_by_keep_two_begin1(feature_tensor):
    """
    每三帧保留前两帧，去掉最后一帧，从第0帧开始
    """
    tokens_per_frame = 196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame

    # Reshape the tensor to (n_frames, tokens_per_frame, -1)
    frames = feature_tensor.view(n_frames, tokens_per_frame, -1)

    # Create a list to store indices of frames to keep
    keep_indices = []

    # Iterate over frames in steps of 3
    for i in range(1, n_frames, 3):
        # Keep frames i, i+1, i+2
        keep_indices.extend(range(i, min(i + 2, n_frames)))

    # Select the frames to keep
    pruned_frames = frames[keep_indices]

    # Reshape the pruned frames back to the original shape
    pruned = pruned_frames.reshape(-1, feature_tensor.size(-1))

    return pruned
def select_random_drop_quarter(feature_tensor):
    tokens_per_frame = 196
    total_tokens = feature_tensor.size(0)
    n_frames = total_tokens // tokens_per_frame  # 计算总帧数

    # 计算需要丢弃的帧数
    frames_to_drop = n_frames // 4

    # 生成所有帧的索引列表
    all_indices = list(range(n_frames))

    # 随机选择需要丢弃的帧索引
    drop_indices = random.sample(all_indices, frames_to_drop)

    # 创建一个掩码，标记需要保留的帧
    keep_indices = [i for i in range(n_frames) if i not in drop_indices]

    # 将特征张量重塑为 (帧数, 每帧token数, 特征维度)
    frames = feature_tensor.view(n_frames, tokens_per_frame, -1)

    # 选择需要保留的帧
    pruned_frames = frames[keep_indices]

    # 将处理后的张量还原为二维形状
    pruned = pruned_frames.reshape(-1, feature_tensor.size(-1))

    return pruned

def select_frames_by_small_cosine_similarity_token_and_video_mean_token_retention_5(input_tensor):
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


def select_frames_by_small_cosine_similarity_token_and_video_mean_token_rentention_0point25(input_tensor):
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
    k_frames = max(1, int(round(0.25 * total_frames)))
    
    # 获取得分最高的k_frames个帧的索引
    _, frame_indices = torch.topk(frame_scores, k=k_frames, largest=False)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(frame_indices, dim=0)
    
    # 收集保留的帧的所有token
    pruned_frames = frames[sorted_indices]
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.reshape(-1, dim)
    
    return pruned_tensor

def select_frames_by_small_cosine_similarity_token_and_video_mean_token_rentention_7(input_tensor):
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
    k_frames = max(1, int(round(0.7 * total_frames)))
    
    # 获取得分最高的k_frames个帧的索引
    _, frame_indices = torch.topk(frame_scores, k=k_frames, largest=False)
    
    # 对索引排序以保持原始帧顺序
    sorted_indices, _ = torch.sort(frame_indices, dim=0)
    
    # 收集保留的帧的所有token
    pruned_frames = frames[sorted_indices]
    
    # 调整形状为 (总保留token数, 896)
    pruned_tensor = pruned_frames.reshape(-1, dim)
    
    return pruned_tensor

def dycoke_ttm_only_half(image_feature, num_tokens_per_frame=196, merging_ratio=0.9):
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio  # Convert to keep ratio

    # Calculate similarities between adjacent frames
    similarities = []
    for i in range(num_frames - 1):
        frame1_tokens = image_feature[i * num_tokens_per_frame : (i+1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i+1) * num_tokens_per_frame : (i+2) * num_tokens_per_frame]
        
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
    
    similarities = torch.stack(similarities)

    # Process adjacent frame pairs (even-odd pruning)
    modified_image_feature = []
    for i in range(0, num_frames - 1, 2):  # Step by 2 to process non-overlapping pairs
        # Get frame pair (i, i+1)
        frame1_tokens = image_feature[i * num_tokens_per_frame : (i+1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i+1) * num_tokens_per_frame : (i+2) * num_tokens_per_frame]

        # Get similarity for this pair (use direct index)
        pair_similarity = similarities[i]
        
        # Calculate tokens to keep from second frame
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = pair_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        # Keep full first frame and pruned second frame
        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(frame2_tokens[tokens_to_keep])

    # Handle odd number of frames (keep last frame as-is)
    if num_frames % 2 != 0:
        last_frame_idx = num_frames - 1
        last_frame = image_feature[last_frame_idx * num_tokens_per_frame : (last_frame_idx+1) * num_tokens_per_frame]
        modified_image_feature.append(last_frame)

    return torch.cat(modified_image_feature, dim=0)


def dycoke_ttm_only_half_use_adjacent_merged_token_replace(image_feature, num_tokens_per_frame=196, merging_ratio=0.7):
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio  # Convert to keep ratio

    # Calculate similarities between adjacent frames
    similarities = []
    for i in range(num_frames - 1):
        frame1_tokens = image_feature[i * num_tokens_per_frame : (i+1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i+1) * num_tokens_per_frame : (i+2) * num_tokens_per_frame]
        
        frame1_norm = torch.nn.functional.normalize(frame1_tokens, p=2, dim=1)
        frame2_norm = torch.nn.functional.normalize(frame2_tokens, p=2, dim=1)
        similarity = torch.nn.functional.cosine_similarity(frame1_norm, frame2_norm, dim=1)
        similarities.append(similarity)
    
    similarities = torch.stack(similarities)

    # Process adjacent frame pairs
    modified_image_feature = []
    for i in range(0, num_frames - 1, 2):
        frame1_tokens = image_feature[i * num_tokens_per_frame : (i+1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i+1) * num_tokens_per_frame : (i+2) * num_tokens_per_frame]

        pair_similarity = similarities[i]
        num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
        tokens_to_keep = pair_similarity.topk(num_tokens_to_keep, largest=False).indices

        next_frame_idx = i + 2
        if next_frame_idx < num_frames:  # 存在后续帧时进行均值替换
            next_frame_tokens = image_feature[next_frame_idx * num_tokens_per_frame : 
                                            (next_frame_idx+1) * num_tokens_per_frame]
            
            # 创建融合后的token
            modified_frame2 = torch.zeros_like(frame2_tokens)
            modified_frame2[tokens_to_keep] = (frame1_tokens[tokens_to_keep] + next_frame_tokens[tokens_to_keep]) / 2
        else:  # 没有后续帧时保留完整i+1帧
            modified_frame2 = frame2_tokens  # 直接使用完整帧，不进行剪枝

        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(modified_frame2)

    # 处理剩余的单数帧
    if num_frames % 2 != 0:
        last_frame = image_feature[-num_tokens_per_frame:]
        modified_image_feature.append(last_frame)

    return torch.cat(modified_image_feature, dim=0)


