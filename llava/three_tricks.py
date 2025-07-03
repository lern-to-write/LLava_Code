def compute_frame_mean_score_matrix_multi_gaussian_small_var_cahnnel(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
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

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=1, keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    return k_xy  # 直接返回核矩阵
def compute_video_mean_score_matrix_multi_gaussian_small_var_channel(input_tensor, token_per_frame=169, alphas=None):
    """使用多高斯核直接计算核矩阵（无需余弦相似度标准化）"""
    if alphas is None:
        alphas = [2**k for k in range(-3, 2)]  # 默认alpha参数
    
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

    frames = torch.nn.functional.normalize(frames, dim=-1)  # 归一化到单位球面
    
    # 计算全局平均token
    avg_token = frames.mean(dim=(0,1), keepdim=True)  # (num_frame, 1, dim)
    
    # 扩展平均token以匹配frames的形状
    expanded_avg = avg_token.expand_as(frames)
    
    # 计算L2距离平方
    l2_distance_square = torch.sum((frames - expanded_avg) ** 2, dim=2)  # (num_frame, token_per_frame)
    
    # 多高斯核：K(x,y) = Σ exp(-||x-y||^2 / (2*alpha))
    k_xy = sum([torch.exp(-l2_distance_square / (2 * alpha)) for alpha in alphas])
    
    return k_xy  # 直接返回核矩阵
