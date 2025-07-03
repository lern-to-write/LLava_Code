import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import torch.nn.functional as F
from torchvision.utils import make_grid
import json
from PIL import Image
from matplotlib.patches import FancyBboxPatch


def remove_lowest_frames(image_feature, frame_scores):
    """
    image_feature: shape [token_number, channel]
    frame_scores: shape [frame_number,]
    returns: new_image_feature with lowest 3 frames removed
    """
    frame_num = frame_scores.size(0)
    token_per_frame = 196  # 根据你的描述设定
    
   # 挑frame ，然后删除
    _, indices_to_remove = torch.topk(frame_scores, k=8, largest=True)
    
    # 创建保留掩码（True表示保留该帧）
    mask = torch.ones(frame_num, dtype=torch.bool, device=frame_scores.device)
    mask[indices_to_remove] = False
    
    # 生成token级别的掩码
    token_mask = mask.repeat_interleave(token_per_frame)
    
    # 应用掩码筛选特征
    new_image_feature = image_feature[token_mask]
    
    return new_image_feature

def visualization_frame_score(frame_score, video_frames, idx):
    """
    video_frames=data.image
    """
    frame_token_length = 196
    frame_number = video_frames.shape[0] 

    # 数据预处理
    reshaped_frames_score = frame_score.reshape(frame_number, 1, 1)
    converted_frames_score = reshaped_frames_score.cpu().detach().numpy().squeeze()

    # 视频帧处理
    video_frames = video_frames.cpu()
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.permute(0, 2, 3, 1).float().cpu().numpy()
    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)

    # 创建画布
    fig = plt.figure(figsize=(20, 3.0))
    gs = fig.add_gridspec(2, 1, 
                        height_ratios=[2.5, 3.5],
                        hspace=-0.2)

    # 计算数据范围和颜色映射
    data_min = converted_frames_score.min()
    data_max = converted_frames_score.max()
    plot_vmax = data_max * 1.1
    norm = plt.Normalize(data_min, data_max)
    cmap = plt.cm.Blues

    # 创建柱状图子图
    ax_bar = fig.add_subplot(gs[0])
    x = np.arange(frame_number)
    scores = converted_frames_score
    
    # 绘制柱状图
    bars = ax_bar.bar(
        x,
        scores - data_min,  # 调整高度使柱子从基线开始
        bottom=data_min,
        width=0.8,
        color=[cmap(norm(score)) for score in scores],
        edgecolor='black',
        linewidth=1,
        alpha=0.9
    )

    # 添加黑色基线（与y=0对齐）
    ax_bar.axhline(y=data_min, color='black', linewidth=1.5, zorder=3)

    ax_bar.plot(x, scores, 
           color='#FFD700',  # 使用更浓郁的金色
           linewidth=2,      # 线条加粗到3
           linestyle='--',
           marker='*',
           markersize=10,
           zorder=4)
    # 设置坐标轴范围和样式
    ax_bar.set_xlim(-0.5, frame_number - 0.5)
    ax_bar.set_ylim(data_min, plot_vmax)
    
    # 完全隐藏y轴
    ax_bar.yaxis.set_visible(False)  # 隐藏整个y轴
    ax_bar.spines['left'].set_visible(False)  # 移除左侧轴线
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)  # 保留底部轴线用于黑色基线
    
    # 隐藏x轴刻度
    ax_bar.tick_params(axis='x', which='both', length=0)
    ax_bar.set_xticks([])

    # 视频帧布局
    gs_video = gs[1].subgridspec(1, frame_number, wspace=0)
    
    # 绘制视频帧
    for i in range(frame_number):
        ax = fig.add_subplot(gs_video[0, i])
        ax.imshow(video_frames[i], aspect='equal')
        ax.set_axis_off()
        ax.margins(0)

    # 保存结果
    save_dir = "/obs/users/yiyu/new_env2/llava/model/image_visualizasion/"
    plt.savefig(os.path.join(save_dir, f'frame_score_barchart{idx}.png'), 
                bbox_inches='tight',
                pad_inches=0.02,
                dpi=300)
    plt.close()



def visualization_each_frames(video_frames):
    """
    video_frames=data.image

    """
    frame_token_length = 196
    frame_number = video_frames.shape[0] 

    # 视频帧处理
    video_frames = video_frames.cpu()
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.permute(0, 2, 3, 1).float().cpu().numpy()
    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)

    # 创建保存目录
    save_dir = '/obs/users/yiyu/new_env2/llava/model/frames/'
    os.makedirs(save_dir, exist_ok=True)

    # 逐帧保存
    for i in range(frame_number):
        # 创建独立画布
        fig = plt.figure(figsize=(4, 3))  # 单帧尺寸可调整
        ax = fig.add_subplot(111)
        
        # 显示单帧图像
        ax.imshow(video_frames[i], aspect='equal')
        ax.set_axis_off()
        ax.margins(0)
        
        # 保存单帧图像
        plt.savefig(os.path.join(save_dir, f'frame_{i}.png'), 
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
                transparent=True)
        plt.close(fig)  # 及时关闭画布释放内存
def mask_non_topk_patches(images, patch_positions, image_size=384, patch_size=27):
    """
    Masks non-top-k patches in images with a translucent white overlay.
    """
   
    num_patches_per_side = image_size // patch_size
    patched_images = images.clone()

    # Define transparency (alpha value)
    alpha = 0.6


    for i, positions in enumerate(patch_positions):
        mask = torch.ones((num_patches_per_side, num_patches_per_side), 
                          dtype=torch.bool, device=images.device)
        
        # 将在positions中的将mask定义为FALSE,不做掩码
        for pos in positions:
            pos=int(pos)
            if not isinstance(pos, int):
                raise TypeError(f"Position {pos} is not an integer.")
            row, col = divmod(pos, num_patches_per_side)
            mask[row, col] = False

        # Apply translucent white mask
        for row in range(num_patches_per_side):
            for col in range(num_patches_per_side):
                start_row = row * patch_size
                end_row = (row + 1) * patch_size
                start_col = col * patch_size
                end_col = (col + 1) * patch_size

                #如果这个地方的mask为true 就进行掩码
                if mask[row, col]:
                    # Create a white overlay

                    white_overlay = torch.ones_like(
                        patched_images[i, :, start_row:end_row, start_col:end_col]
                    )
                    patched_images[i, :, start_row:end_row, start_col:end_col] = (
                        alpha * white_overlay + 
                        (1 - alpha) * patched_images[i, :, start_row:end_row, start_col:end_col]
                    )
    
    return patched_images

def mask_selected_frames(images, frame_indices, alpha=0.6):
    """
    Masks entire frames specified by frame_indices with a translucent white overlay.
    
    Args:
        images (Tensor): Input image tensor of shape (B, C, H, W)
        frame_indices (list[int]): List of frame indices to be masked
        alpha (float): Transparency value (0.0 = fully transparent, 1.0 = fully opaque)
    
    Returns:
        Tensor: Masked image tensor with the same shape as input
    """
    masked_images = images.clone()
    
    # Validate frame indices
    valid_indices = [idx for idx in frame_indices if 0 <= idx < images.size(0)]
    if len(valid_indices) != len(frame_indices):
        invalid = set(frame_indices) - set(valid_indices)
        print(f"Warning: Invalid frame indices {invalid} were ignored")

    # Create white overlay (shape: [1, C, H, W] for broadcasting)
    white_overlay = torch.ones(
        (1, images.size(1), images.size(2), images.size(3)),
        dtype=images.dtype,
        device=images.device
    )

    # Apply mask to selected frames
    for idx in valid_indices:
        masked_images[idx] = alpha * white_overlay + (1 - alpha) * masked_images[idx]
    
    return masked_images
def arrange_and_save_images(original_images, patched_images, idx):
    """
    Arranges original and patched images into a grid and saves them.

    """
    save_dir = '/root/autodl-tmp/LLaVA-NeXT/llava/model/patch_position_visualizasion_25'

    os.makedirs(save_dir, exist_ok=True)
    assert original_images.shape[0] == patched_images.shape[0], \
        "Original and patched images must have the same number of images."
    num_images = original_images.shape[0]
    
    arranged_images = torch.zeros((2 * num_images, *original_images.shape[1:]), 
                                  dtype=original_images.dtype)
    arranged_images[:num_images] = original_images.cpu()
    arranged_images[num_images:] = patched_images.cpu()
    
    grid = make_grid(arranged_images, nrow=num_images, padding=2, normalize=True)
    save_image(grid, os.path.join(save_dir, f'arranged_images{idx}.png'))


def compute_masked_mean(attention_tensor, threshold= 0.0000e+00):
    # 创建掩码，标记绝对值超过阈值的有效元素
    mask = (attention_tensor > threshold).float()
    
    # 计算每行有效元素的和及数量
    sum_per_row = torch.sum(attention_tensor * mask, dim=1)
    count_per_row = torch.sum(mask, dim=1)
    
    # 避免除零，添加极小值
    count_per_row = count_per_row + 1e-12
    
    # 计算均值并返回
    return sum_per_row / count_per_row
    

def compute_and_plot_attention(hidden_states, system_token_length, all_image_token_length, frame_number, frame_token_length):
    '''
    绘制折线图
    '''
    # 初始化 attention_for_frame_list
    attention_for_frame_list = []

    # 提取 video_tensor
    video_tensor = hidden_states[0][system_token_length:system_token_length + all_image_token_length].view(frame_number, frame_token_length, -1).sum(-1)

    # 计算每一帧之间的 L1 距离，并将结果存入 attention_for_frame_list
    for i in range(video_tensor.size(0) - 1):
        frame_i = video_tensor[i]
        frame_j = video_tensor[i+1]
        l1_dist = torch.dist(frame_i, frame_j, p=2)  # 使用 L2 距离（可以根据需要调整为 L1）
        attention_for_frame_list.append(l1_dist)
        print(f"Frame {i} to Frame {i+1} L2 Distance: {l1_dist.item()}")

    # 打印 attention_for_frame_list
    print("attention_for_frame_list:", attention_for_frame_list)

    # 将 attention_for_frame_list 转换为普通的 Python 列表
    attention_values = [tensor.item() for tensor in attention_for_frame_list]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(attention_values, marker='o')
    plt.title('Attention Values Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Attention Value')
    plt.grid(True)

    # 保存图像
    output_file = '/root/autodl-tmp/LLaVA-NeXT/llava/attention_values_over_frames.png'
    plt.savefig(output_file)
    plt.close()



def fastv_visualize_attention(multihead_attention, output_path="atten_map_1.png", title="Layer 2"):
    '''
    fastv可视化三角形注意力图
    '''
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].mean(dim=-1).view(16,196).float()  # 形状变为 (n_tokens, n_tokens)
    
    cmap = plt.cm.get_cmap("viridis")  # 选择viridis颜色映射
    plt.figure(figsize=(5, 5), dpi=400)  # 创建5x5英寸，400DPI的高清图像
    
    # 创建对数归一化器（用于显示微小值差异）
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    
    # 创建热力图
    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,        # 使用指定颜色映射
        norm=log_norm,    # 应用对数归一化
        # cbar_kws={'label': 'Attention score'},  # （注释掉的）颜色条标签
    )
    
    # 设置坐标轴刻度标签（原始token位置的20倍）
    n_ticks = averaged_attention.shape[0]
    tick_labels = [str(i*20) for i in range(n_ticks)]
    ax.set_xticks(range(n_ticks))      # 设置刻度位置为0,1,2...
    ax.set_yticks(range(n_ticks))
    ax.set_xticklabels(tick_labels)    # 标签显示实际token位置
    ax.set_yticklabels(tick_labels)
    
    plt.xticks(fontsize=3)  # 缩小x轴标签字体
    plt.yticks(fontsize=3)  # 缩小y轴标签字体
    plt.yticks(rotation=0)  # y标签水平显示
    plt.xticks(rotation=90) # x标签垂直显示
    
    plt.title(title)  # 添加标题
    plt.savefig(output_path, bbox_inches='tight')  # 保存紧凑布局的图片
    
    # 提取每行的前10个注意力峰值（虽然变量名是top_five）
    top_five_attentions = []
    for row in averaged_attention:
        top_values, top_indices = torch.topk(row, 10)  # 获取前10大值和索引
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions, averaged_attention  # 返回分析结果和平均注意力矩阵

def visualization_attention_map(layer_outputs, video_frames, system_token_length, all_image_token_length, frame_number, frame_token_length):

    """ 
    video_frames=data.image

    """
    # 获取最后一层注意力
    last_layer_attention = layer_outputs[1]
    attention_map = last_layer_attention.mean(dim=1)[0][-1][system_token_length:system_token_length+all_image_token_length].view(16, 196)

    # 对每个帧进行对数归一化
    attention_map_log = torch.log(attention_map + 1e-8)  # 防止对数零值
    attention_map_log = (attention_map_log - attention_map_log.min(dim=1, keepdim=True)[0]) / \
                        (attention_map_log.max(dim=1, keepdim=True)[0] - attention_map_log.min(dim=1, keepdim=True)[0] + 1e-8)

    # 将注意力图转换为帧
    
    video_frames = video_frames.cpu()
    num_frames = frame_number
    num_patches = frame_token_length
    patch_size = int(np.sqrt(num_patches))

    frames = np.zeros((num_frames, patch_size, patch_size))
    for i in range(num_frames):
        for j in range(num_patches):
            row, col = divmod(j, patch_size)
            frames[i, row, col] = attention_map_log[i, j]

    # 归一化视频帧
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.permute(0, 2, 3, 1).float().cpu().numpy()
    else:
        video_frames = video_frames.transpose(0, 2, 3, 1)

    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)
    frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)

    # 绘制结果
    fig, axes = plt.subplots(2, num_frames, figsize=(20, 4), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    for i in range(num_frames):
        axes[0, i].imshow(video_frames[i], aspect='equal')
        axes[0, i].axis('off')

    vmin, vmax = frames.min(), frames.max()
    for i in range(num_frames):
        im = axes[1, i].imshow(frames[i], cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig('combined_attention.png', bbox_inches='tight', dpi=300)
    plt.close()


def visualization_siglip_attention_map(attention_map, video_frames):

    """
    video_frames=data.image
    """
    frame_token_length=196
    frame_number=16
    # 对每个帧进行对数归一化
    # attention_map_log = torch.log(attention_map + 1e-8)  # 防止对数零值
    # attention_map_log = (attention_map_log - attention_map_log.min(dim=1, keepdim=True)[0]) / \
    #                     (attention_map_log.max(dim=1, keepdim=True)[0] - attention_map_log.min(dim=1, keepdim=True)[0] + 1e-8)
    attention_map_log=attention_map
    # 将注意力图转换为帧
    video_frames = video_frames.cpu()
    num_frames = frame_number
    num_patches = frame_token_length
    patch_size = int(np.sqrt(num_patches))

    # frames = np.zeros((num_frames, patch_size, patch_size))
    # for i in range(num_frames):
    #     for j in range(num_patches):
    #         row, col = divmod(j, patch_size)
    #         frames[i, row, col] = attention_map_log[i, j]
    frames = attention_map_log.cpu().detach().numpy().reshape(16,patch_size, patch_size)
    # 归一化视频帧
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.permute(0, 2, 3, 1).float().cpu().numpy()
    else:
        video_frames = video_frames.transpose(0, 2, 3, 1)
    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)
    fig, axes = plt.subplots(2, num_frames, figsize=(20, 4), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for i in range(num_frames):
        axes[0, i].imshow(video_frames[i], aspect='equal')
        axes[0, i].axis('off')
    vmin, vmax = frames.min(), frames.max()
    for i in range(num_frames):
        im = axes[1, i].imshow(frames[i], cmap='viridis', aspect='equal', norm=plt.Normalize(np.min(frames), np.max(frames)),)
        axes[1, i].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('siglip_attention.png', bbox_inches='tight', dpi=300)
    plt.close()




def visualization_cosin_similarity_score(video_frames, num_frames, attn_scores):
    """
    在每个patch底下显示分数
    """
    # 转换张量格式并归一化
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.permute(0, 2, 3, 1).float().cpu().numpy()
    else:
        video_frames = video_frames.transpose(0, 2, 3, 1)
        
    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)
    
    # 获取视频参数
    T, H, W, C = video_frames.shape
    num_frames = min(num_frames, T)
    patch_size = 27
    
    # 计算分割参数
    n_rows = H // patch_size
    n_cols = W // patch_size
    
    # 创建画布（调整整体布局）
    fig = plt.figure(figsize=(n_cols*num_frames*2, n_rows*3))  # 增加垂直空间
    outer_gs = fig.add_gridspec(1, num_frames, wspace=0.1, hspace=0.1)

    # 遍历每个帧
    for frame_idx in range(num_frames):
        frame = video_frames[frame_idx]
        
        # 创建内部网格（增加行间距）
        inner_gs = outer_gs[frame_idx].subgridspec(n_rows, n_cols, 
                                                 wspace=0.1, hspace=0.5)  # 增大hspace
        
        # 遍历每个补丁
        for row in range(n_rows):
            for col in range(n_cols):
                # 计算补丁坐标
                y_start = row * patch_size
                y_end = y_start + patch_size
                x_start = col * patch_size
                x_end = x_start + patch_size
                
                # 创建子图并显示补丁
                ax = fig.add_subplot(inner_gs[row, col])
                ax.imshow(frame[y_start:y_end, x_start:x_end], aspect='auto')
                ax.axis('off')
                
                # 添加注意力分数（在子图下方显示）
                score = attn_scores[frame_idx, row, col]
                ax.text(0.5,                       # x居中
                        -0.2,                      # y位置在子图下方
                        f"{score:.2f}",            # 格式化分数
                        ha='center', va='top',     # 对齐方式
                        transform=ax.transAxes,    # 使用轴坐标系
                        fontsize=8,                # 调整字体大小
                        color='white',             # 字体颜色
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))  # 背景框

    plt.savefig('frame_mean.png', bbox_inches='tight', dpi=300)
    plt.close()


def process_and_mask_features(input_tensor, original_images,keep_ratio=0.5):
    """
    根据frame mean cosine 进行mask
    """
    # 计算总帧数并重塑张量
    num_frames = input_tensor.shape[0] // 196
    frames = input_tensor.view(num_frames, 196, -1)
    _, tokens_per_frame, _ = frames.shape
    
    # 计算每帧平均特征
    avg_per_frame = frames.mean(dim=1, keepdim=True)
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(
        frames,
        avg_per_frame.expand_as(frames),
        dim=2
    )
    
    # 计算保留token数并获取索引
    k = int(round(keep_ratio * tokens_per_frame))

    _, indices = torch.topk(similarities, k=k, dim=1, largest=False, sorted=False)
    
    # 排序索引并转换为列表格式
    sorted_indices, _ = torch.sort(indices, dim=1)
    sorted_indices_list = [frame_indices.tolist() for frame_indices in torch.unbind(sorted_indices, dim=0)]
    
    # 生成并保存mask后的图像
    patched_images = mask_non_topk_patches(original_images, sorted_indices_list)
    arrange_and_save_images(original_images, patched_images)
    
    return patched_images


def process_and_mask_features_score(input_tensor, sorted_indices_list,original_images):
    """
    生成score 的mask
    """
    
    
    # 生成并保存mask后的图像
    patched_images = mask_non_topk_patches(original_images, sorted_indices_list)
    arrange_and_save_images(original_images, patched_images)
    
    return patched_images


# 计数器文件路径
COUNTER_FILE = "video_counter.json"

# 初始化或读取计数器
def get_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            return json.load(f)['count']
    else:
        return 0  # 初始值

def update_counter():
    current = get_counter() + 1
    with open(COUNTER_FILE, 'w') as f:
        json.dump({'count': current}, f)
    return current
