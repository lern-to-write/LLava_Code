import torch

def dpc_knn_cluster(features, num_clusters):
    """
    使用DPC-KNN算法进行聚类
    :param features: 待聚类特征 (n_samples, dim)
    :param num_clusters: 目标聚类数量
    :return: 聚类中心索引 (num_clusters,)
    """
    n_samples = features.shape[0]
    if n_samples <= num_clusters:
        return torch.arange(n_samples, device=features.device)

    # 计算距离矩阵
    distances = torch.cdist(features, features, p=2)
    
    # 计算KNN密度（使用k=5）
    k = min(5, n_samples-1)
    # knn_dists, _ = torch.topk(distances, k=k+1, dim=1, largest=False, sorted=True)
    # local_density = 1.0 / (knn_dists[:, 1:].mean(dim=1))  # 排除自身

    dist_sq = distances**2  # 显式计算平方距离
    knn_dists_sq, _ = torch.topk(dist_sq, k=k+1, dim=1, largest=False)
    local_density = torch.exp(-knn_dists_sq[:, 1:].mean(dim=1))
    
    # 计算相对距离
    density_order = torch.argsort(local_density, descending=True)
    delta = torch.zeros_like(local_density)
    nearest_higher = torch.zeros(n_samples, dtype=torch.long, device=features.device)
    
    for i, idx in enumerate(density_order):
        higher_density = density_order[:i]
        if len(higher_density) == 0:
            delta[idx] = distances[idx].max()
            nearest_higher[idx] = -1
        else:
            min_dist, min_idx = distances[idx, higher_density].min(dim=0)
            delta[idx] = min_dist
            nearest_higher[idx] = higher_density[min_idx]
    
    # 选择聚类中心
    scores = local_density * delta
    _, centers = torch.topk(scores, k=num_clusters)
    return centers

def get_indeice_of_select_token_with_double_specific_retention_big_merge_and_small_select(
    input_tensor, 
    scales, 
    video_mean_score,
    frame_mean_score, 
    token_per_frame=169
):
    # 输入校验
    assert len(input_tensor.shape) == 2, "Input tensor应为二维 (num_tokens, dim)"
    num_frame = input_tensor.shape[0] // token_per_frame
    assert len(scales) == num_frame, "scales长度必须等于帧数"
    
    # 设备信息
    device = input_tensor.device
    
    # 预处理输入
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    combined_similarities = video_mean_score + frame_mean_score
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
    
    pruned_indices = []
    for frame_idx in range(num_frame):
        # 当前帧数据
        frame = frames[frame_idx]
        sim = combined_similarities[frame_idx]
        k = max(1, int(round(scales[frame_idx].item() * token_per_frame)))
        
        # 计算保留比例
        k_low =  int(k*0.7)   # 保留低分token数量
        k_merge = int(k - k_low)    # 需要聚类的数量
        
        # 阶段1：选择低分token
        _, low_indices = torch.topk(sim, k=k_low, largest=False, sorted=False)
        
        # 阶段2：处理需要聚类的token
        all_indices = torch.arange(token_per_frame, device=device)
        mask = torch.ones(token_per_frame, dtype=torch.bool, device=device)
        mask[low_indices] = False
        remaining_indices = all_indices[mask]
        
        merge_indices = torch.tensor([], dtype=torch.long, device=device)
        if k_merge > 0 and len(remaining_indices) > 0:
            # 执行DPC-KNN聚类
            remaining_features = frame[remaining_indices]
            cluster_centers = dpc_knn_cluster(remaining_features, k_merge)
            
            # 处理聚类结果
            valid_clusters = min(len(cluster_centers), k_merge)
            merge_indices = remaining_indices[cluster_centers[:valid_clusters]]
        
        # 合并并排序索引
        combined = torch.cat([low_indices, merge_indices])
        combined = torch.unique(combined)  # 去重（当剩余token不足时）
        sorted_indices, _ = torch.sort(combined)
        
        # 截断确保数量不超过k
        pruned_indices.append(sorted_indices[:k])
    
    return pruned_indices



def get_indeice_of_select_token_with_double_specific_retention_big_merge_and_small_select_30(
    input_tensor, 
    scales, 
    video_mean_score,
    frame_mean_score, 
    token_per_frame=169
):
    # 输入校验
    assert len(input_tensor.shape) == 2, "Input tensor应为二维 (num_tokens, dim)"
    num_frame = input_tensor.shape[0] // token_per_frame
    assert len(scales) == num_frame, "scales长度必须等于帧数"
    
    # 设备信息
    device = input_tensor.device
    
    # 预处理输入
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    combined_similarities = video_mean_score + frame_mean_score
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
    
    pruned_indices = []
    for frame_idx in range(num_frame):
        # 当前帧数据
        frame = frames[frame_idx]
        sim = combined_similarities[frame_idx]
        k = max(1, int(round(scales[frame_idx].item() * token_per_frame)))
        
        # 计算保留比例
        k_low =  int(k*0.7)   # 保留低分token数量
        k_high=int(token_per_frame*0.3)
        k_merge = int(k - k_low)    # 需要聚类的数量
        
        # 阶段1：选择低分token
        _, low_indices = torch.topk(sim, k=k_low, largest=False, sorted=False)
        _, high_indices = torch.topk(sim, k=k_high, largest=True, sorted=False)

        
        # 阶段2：处理需要聚类的token
        all_indices = torch.arange(token_per_frame, device=device)
        mask = torch.zeros(token_per_frame, dtype=torch.bool, device=device)
        mask[high_indices] = True
        remaining_indices = all_indices[mask]
        
        merge_indices = torch.tensor([], dtype=torch.long, device=device)
        if k_merge > 0 and len(remaining_indices) > 0:
            # 执行DPC-KNN聚类
            remaining_features = frame[remaining_indices]
            cluster_centers = dpc_knn_cluster(remaining_features, k_merge)
            
            # 处理聚类结果
            valid_clusters = min(len(cluster_centers), k_merge)
            merge_indices = remaining_indices[cluster_centers[:valid_clusters]]
        
        # 合并并排序索引
        combined = torch.cat([low_indices, merge_indices])
        combined = torch.unique(combined)  # 去重（当剩余token不足时）
        sorted_indices, _ = torch.sort(combined)
        
        # 截断确保数量不超过k
        pruned_indices.append(sorted_indices[:k])
    
    return pruned_indices






def get_indeice_of_select_token_with_single_specific_retention_big_merge_and_small_select(
    input_tensor, 
    scales, 
    frame_mean_score, 
    token_per_frame=169
):
    # 输入校验
    assert len(input_tensor.shape) == 2, "Input tensor应为二维 (num_tokens, dim)"
    num_frame = input_tensor.shape[0] // token_per_frame
    assert len(scales) == num_frame, "scales长度必须等于帧数"
    
    # 设备信息
    device = input_tensor.device
    
    # 预处理输入
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    combined_similarities =  frame_mean_score
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
    
    pruned_indices = []
    for frame_idx in range(num_frame):
        # 当前帧数据
        frame = frames[frame_idx]
        sim = combined_similarities[frame_idx]
        k = max(1, int(round(scales[frame_idx].item() * token_per_frame)))
        
        # 计算保留比例
        k_low =  int(k*0.7 )  # 保留低分token数量
        k_merge = int(k - k_low)    # 需要聚类的数量
        
        # 阶段1：选择低分token
        _, low_indices = torch.topk(sim, k=k_low, largest=False, sorted=False)
        
        # 阶段2：处理需要聚类的token
        all_indices = torch.arange(token_per_frame, device=device)
        mask = torch.ones(token_per_frame, dtype=torch.bool, device=device)
        mask[low_indices] = False
        remaining_indices = all_indices[mask]
        
        merge_indices = torch.tensor([], dtype=torch.long, device=device)
        if k_merge > 0 and len(remaining_indices) > 0:
            # 执行DPC-KNN聚类
            remaining_features = frame[remaining_indices]
            cluster_centers = dpc_knn_cluster(remaining_features, k_merge)
            
            # 处理聚类结果
            valid_clusters = min(len(cluster_centers), k_merge)
            merge_indices = remaining_indices[cluster_centers[:valid_clusters]]
        
        # 合并并排序索引
        combined = torch.cat([low_indices, merge_indices])
        combined = torch.unique(combined)  # 去重（当剩余token不足时）
        sorted_indices, _ = torch.sort(combined)
        
        # 截断确保数量不超过k
        pruned_indices.append(sorted_indices[:k])
    
    return pruned_indices



def get_indeice_of_select_token_with_single_specific_retention_big_merge_and_small_select_30(
    input_tensor, 
    scales, 
    frame_mean_score, 
    token_per_frame=169
):
    # 输入校验
    assert len(input_tensor.shape) == 2, "Input tensor应为二维 (num_tokens, dim)"
    num_frame = input_tensor.shape[0] // token_per_frame
    assert len(scales) == num_frame, "scales长度必须等于帧数"
    
    # 设备信息
    device = input_tensor.device
    
    # 预处理输入
    frames = input_tensor.view(num_frame, token_per_frame, -1)
    combined_similarities =  frame_mean_score
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
    
    pruned_indices = []
    for frame_idx in range(num_frame):
        # 当前帧数据
        frame = frames[frame_idx]
        sim = combined_similarities[frame_idx]
        k = max(1, int(round(scales[frame_idx].item() * token_per_frame)))
        
        # 计算保留比例
        k_low =  int(k*0.7 )  # 保留低分token数量
        k_high=int(token_per_frame*0.3)
        k_merge =int( k - k_low)    # 需要聚类的数量
        
        # 阶段1：选择低分token
        _, low_indices = torch.topk(sim, k=k_low, largest=False, sorted=False)
        _, high_indices = torch.topk(sim, k=k_high, largest=False, sorted=False)

        
        # 阶段2：处理需要聚类的token
        all_indices = torch.arange(token_per_frame, device=device)
        mask = torch.zeros(token_per_frame, dtype=torch.bool, device=device)
        mask[high_indices] = True
        remaining_indices = all_indices[mask]
        
        merge_indices = torch.tensor([], dtype=torch.long, device=device)
        if k_merge > 0 and len(remaining_indices) > 0:
            # 执行DPC-KNN聚类
            remaining_features = frame[remaining_indices]
            cluster_centers = dpc_knn_cluster(remaining_features, k_merge)
            
            # 处理聚类结果
            valid_clusters = min(len(cluster_centers), k_merge)
            merge_indices = remaining_indices[cluster_centers[:valid_clusters]]
        
        # 合并并排序索引
        combined = torch.cat([low_indices, merge_indices])
        combined = torch.unique(combined)  # 去重（当剩余token不足时）
        sorted_indices, _ = torch.sort(combined)
        
        # 截断确保数量不超过k
        pruned_indices.append(sorted_indices[:k])
    
    return pruned_indices