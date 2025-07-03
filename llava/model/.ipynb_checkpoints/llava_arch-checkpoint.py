#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import pdb  # 导入 pdb 模块
from llava.model.language_model.devide_clips import *
from llava.model.frame_feature_selection import *
from llava.model.visualization import *
# from llava.model.frame_merge import *
# from llava.model.test_base_frame import *
from llava.model.token_select import *
from llava.model.token_select import *
from llava.model.frame_select_before_vit import *
from llava.model.globle_to_local import *
import torch.nn.functional as F
from llava.model.data_store import data
# import logging
import os

# # 配置日志记录器
# logging.basicConfig(
#     filename='/root/autodl-tmp/LLaVA-NeXT/llava/model/llava_arch.log',  # 日志文件路径
#     level=logging.DEBUG,         # 日志级别
#     format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
# )

# # logger = logging.get_logger(__name__)
# logger = logging.getLogger(__name__)

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        #__init__ 方法初始化 LlavaMetaModel 类的实例。
        #检查 config 是否包含 mm_vision_tower 属性，如果有，则初始化视觉塔、视觉重采样器和多模态投影器。
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
        
            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
   #这段代码主要用于初始化和配置多模态模型的视觉模块。它根据提供的配置和参数构建视觉塔、视觉重采样器和多模态投影器，并加载相应的预训练权重
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
  
        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
        #如果视觉塔未初始化，则构建视觉塔和视觉重采样器，并将其配置添加到 config 中。
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)
            #如果 fsdp 不为空且长度大于 0，则将视觉塔和视觉重采样器存储在列表中。

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower

            #载视觉塔模型，并确保视觉重采样器的参数可训练。    
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        #如果 config 中没有 add_faster_video 属性且 model_args 中设置了 add_faster_video，则初始化 faster_token 参数。
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        #如果多模态投影器未初始化，则构建多模态投影器，并根据补丁合并类型初始化 image_newline 参数。
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
       #如果提供了预训练适配器，则加载多模态投影器和视觉重采样器的权重，并打印不兼容的键。
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    #get_2dPool 方法对图像特征进行 2D 池化操作，根据配置选择平均池化、最大池化或双线性插值。
    def get_2dPool(self, image_feature, stride=2):
        #(Pdb) image_feature.shape
        #torch.Size([16, 729, 3584])

        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    #encode_images 方法使用视觉塔和多模态投影器对图像进行编码。
    def encode_images(self, images):
        #(Pdb) images.shape
        #torch.Size([16, 3, 384, 384])

        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        
        #16 729 3584
        return image_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):

        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)

        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  

        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            #检测是慢速视频帧还是快速视频帧
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        
        return all_videos_or_images_features,all_faster_video_features
    #add_token_per_grid 方法在每个网格中添加标记
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature
    #add_token_per_frame 方法在每帧中添加标记。
    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        
        #image 1 16 3 384 384   
        # (B, N, C, H, W)
        # input_ids  1 29
        
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        #如果视觉塔或图像为空，或者输入 ID 的形状为 1，则直接返回原始输入。
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        #如果模态类型是字符串，则将其转换为列表。
        if isinstance(modalities, str):
            modalities = [modalities]

        #如果图像数据是列表或 5 维张量，则将其转换为 4 维张量。
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                #pdb.set_trace()
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            video_idx_in_batch = []
            #获取视频索引：遍历模态类型，获取视频在batch中的索引。
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)
            #将图像拼接成一个张量，调用 self.encode_images(concat_images) 对图像进行编码，并根据原始图像的大小拆分编码后的特征。
            images_list = []
            
            for image in images:
                #images 16 3 384 384
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    #如果是单图，增加一个维度
                    images_list.append(image.unsqueeze(0))
            #这里其实就是把列表转换成张量
            concat_images = torch.cat([image for image in images_list], dim=0)# 16*3*384*384

            #### visualazation #########
            data.image=concat_images
            
            #### visualization end #########
            # concat_images=prune_frames_before_vit(concat_images)
            ######################################### auto start ##########################################

            # def dynamic_processor(concat_images):
            #     processor_type = os.getenv("PROCESSOR", "prune_tokens")
                
            #     # 定义可切换的函数集合
            #     processors = {
            #         "select_frames_based_on_50percent_scores_sum":select_frames_based_on_50percent_scores_sum(concat_images),
            #         "select_frames_based_on_30percent_scores_sum":select_frames_based_on_30percent_scores_sum(concat_images),
            #         "select_frames_based_on_50percent_scores_channel":select_frames_based_on_50percent_scores_channel(concat_images),
            #         "select_frames_based_on_30percent_scores_chanenel":select_frames_based_on_30percent_scores_chanenel(concat_images),

     
            #     }
            #     return processors[processor_type]

            # concat_images = dynamic_processor(concat_images)

            # l1_scores=compute_global_l1_scores_max_pool(concat_images)


            ######################################### auto end ##########################################
            # split_sizes = [image.shape[0] for image in images_list]

            split_sizes = [concat_images.shape[0]]
            encoded_image_features = self.encode_images(concat_images)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)

            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            #根据配置处理patch合并类型，如果 mm_patch_merge_type 为 "flat"，则将图像特征平坦化
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            #如果 mm_patch_merge_type 以 "spatial" 开头，则处理空间合并类型。
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):

                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    
                    #如果当前图像索引在 video_idx_in_batch 中，则进行视频特征处理。
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        #如果 mm_newline_position 为 "grid"，则调用 self.add_token_per_grid 方法添加网格标记。如果配置中启用了 add_faster_video，则进一步处理视频特征。
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                
                                image_feature = torch.cat(concat_slow_fater_token)
                        
                            new_image_features.append(image_feature)
                            #如果 mm_newline_position 为 "frame"，则调用 self.add_token_per_frame 方法添加帧标记，并将特征展平
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        #如果 mm_newline_position 为 "one_token"，则将特征展平，并根据配置添加新行标记。    
                        elif mm_newline_position == "one_token":
                            # one-token
                            #(a, b, c)-> (a*b, c)
                            #image_feature.shape([16, 196, 3584])
  
                            image_feature = image_feature.flatten(0, 1)#([3136, 3584])
                            
                            if 'unpad' in mm_patch_merge_type:
                                #########################################################################################
                                ############################# auto running#############################################
                                # def dynamic_processor(sumed_key,sumed_value):
                                #     processor_type = os.getenv("PROCESSOR", "prune_tokens")
                                    
                                #     # 定义可切换的函数集合
                                #     processors = {

                                #         # "frame_score_by_big_attn_video_mean":frame_score_by_big_attn_video_mean(interpolate_features,video_mean_score),
                                #         # "frame_score_by_small_attn_video_mean":frame_score_by_small_attn_video_mean(interpolate_features,video_mean_score),
                                #         "compute_frame_score_top_10_percent_interpolate_key_negative":compute_frame_score_top_10_percent_interpolate_key_negative(sumed_key),
                                #         "compute_frame_score_top_10_percent_interpolate_key_positive":compute_frame_score_top_10_percent_interpolate_key_positive(sumed_key),
                                #         "compute_frame_score_full_interpolate_key_negative":compute_frame_score_full_interpolate_key_negative(sumed_key),
                                #         "compute_frame_score_full_interpolate_key_positive":compute_frame_score_full_interpolate_key_positive(sumed_key),

                                #         "compute_frame_score_top_10_percent_interpolate_value_negative":compute_frame_score_top_10_percent_interpolate_key_negative(sumed_value),
                                #         "compute_frame_score_top_10_percent_interpolate_value_positive":compute_frame_score_top_10_percent_interpolate_key_positive(sumed_value),
                                #         "compute_frame_score_full_interpolate_value_negative":compute_frame_score_full_interpolate_key_negative(sumed_value),
                                #         "compute_frame_score_full_interpolate_value_positive":compute_frame_score_full_interpolate_key_positive(sumed_value),
        
                                      
                                #     }
                                #     return processors[processor_type]

                                
                                
                                ############################# auto running ##########################################\
                                interpolate_value=data.interpolate_value

                                interpolate_features=data.interpolate_features
                            
                                interpolate_key=data.interpolate_key
                                # key_norms = torch.norm(interpolate_key, p=2, dim=-1)
                                # value_norms = torch.norm(interpolate_value, p=2, dim=-1)

                                flattened_key = interpolate_key.view(-1, interpolate_key.shape[2])
                                # video_mean_score=compute_video_mean_score(image_feature)
                                # frame_mean_score=compute_frame_mean_score(image_feature)
                                var_video_mean_score=compute_video_mean_score_high_var(flattened_key)
                                var_frame_mean_score=compute_frame_mean_score_high_var(flattened_key)
                                # frame_mean_score=compute_frame_mean_score_high_var(image_feature)
                                # scales=generate_scales_whith_mean_video_score(image_feature)
                                num_frames=image_feature.shape[0]//196
                                scales = torch.full((num_frames,), 0.3)
                                # frame_score = compute_frame_score_top_10_percent_interpolate_features_negative(interpolate_features)
                                #####################################################################                                 
                                # scales=generate_scales_from_frame_video_score(frame_score)
                                ################################################################

                                # image_feature=select_token_with_single_specific_retention(image_feature, scales,var_frame_mean_score)
                                image_feature=select_token_with_conbine_specific_retention(image_feature, scales,var_video_mean_score,var_frame_mean_score)
                                # image_feature=random_drop_tokens(image_feature)
                                # image_feature=select_token_with_mixed_retention(image_feature, scales,var_frame_mean_score)







                                # ################################################
                                # scales=generate_scales_whith_mean_video_score(image_feature,base_scale=0.5,temperature=0.1)
                                # image_feature=select_token_with_frame_specific_retention(image_feature, scales)




                                #################### 可视化 #############################################
                                # base_frames_index,l1_distance_list=select_base_frames_by_neighbor_l1(image_feature)
                                # # base_frames_index=data.frame_index
                                # patched_images=mask_selected_frames(original_image_feature, base_frames_index)
                                # arrange_and_save_images(original_image_feature, patched_images, save_dir='output_images')
                                





            
    

                                    

                                ######################################################################################
                                # data.image_feature_lenth=image_feature.shape[0]

                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                                #image_feature.shape([3137, 3584])
                            
                            new_image_features.append(image_feature)
                            
                        #如果 mm_newline_position 为 "no_token"，则直接将特征展平。    
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

                        #pdb.set_trace()
                        #如果图像特征的第一个维度大于 1，则进行多补丁和多图像操作，
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations

                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        #如果 image_aspect_ratio 包含 "anyres_max"，则匹配最大补丁数。
                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)
                        #：如果 mm_patch_merge_type 包含 "maxpool2x2"，则对图像特征进行最大池化。
                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        #如果 mm_patch_merge_type 包含 "unpad"，则对图像特征进行去填充处理。
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            #多图在这里运行
                            #pdb.set_trace()# image_feature.shape torch.Size([3, 3, 27, 27, 3584])
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            #image_feature.shape  torch.Size([3584, 71, 81])
                            
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            #multi_image after unpad torch.Size([5822, 3584])
                        elif "unpad" in mm_patch_merge_type:
                            
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            #如果 mm_patch_merge_type 不包含 "nobase"，则将基础特征与处理后的特征拼接。
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                        #final image feature  torch.Size([6551, 3584])
                    # single image operations
                    else:  

                        image_feature = image_feature[0]
                        #如果 mm_patch_merge_type 包含 "unpad"，则对图像特征进行去填充处理。
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)

                image_features = new_image_features
                #image_feature   torch.Size([3137, 3584])
                #new_image_features[0][6551,896] new_image_features[1].shapetorch.Size([3699, 896])
                
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        image_token_indices_all = [] 
        image_token_indices_in_batches=[]
        # rank_print("Inserting Images embedding")

        #cur_image_features图像特征
        #cur_input_ids 文本token
        #cur_input_embeds把这两个拼在一起
        #logging.debug(f"input_ids: {input_ids}")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices_in_batches=torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            image_token_indices = [-1] + image_token_indices_in_batches.tolist() + [cur_input_ids.shape[0]]           
            #logging.debug(f"image_token_indices: {image_token_indices}")

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            #将非图像token重新拆分
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)#返回元组或者列表
            cur_new_input_embeds = []
            cur_new_labels = []

            #图像和文本交替嵌入
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            ########### frame wise bigen ################
            
            all_image_token_length= cur_new_input_embeds[1].shape[0]
            frame_token_length=196
            frame_number=all_image_token_length/196
            system_token_length=cur_new_input_embeds[0].shape[0]
            user_instruction_length=cur_new_input_embeds[2].shape[0]
            data.prepare(all_image_token_length, frame_token_length, frame_number, system_token_length,user_instruction_length)

            ############### frame wise end ################
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            #torch.Size([6578, 896])
           
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)# cur_new_input_embeds torch.Size([6578, 896])

            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_token_indices_all.append(image_token_indices_in_batches)  
                    
            #logging.debug(f"batch_idx{batch_idx}cur_new_input_embeds: {cur_new_input_embeds.shape}")

        image_token_indices_all = torch.stack(image_token_indices_all, dim=0)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")
        
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #torch.Size([6578, 896])
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")
        #处理成统一形状
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)


        # logging.debug(f"new_input_embeds {new_input_embeds.shape}")
        # logging.debug(f"new_input_embeds_padded: {new_input_embeds_padded[0].shape}")
        # logging.debug(f"new_input_embeds {new_input_embeds.shape}")
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        ### FRAMEFUSION START ###

        ### FRAMEFUSION END ###
        return None,position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
# video new_input_embeds.shape# torch.Size([1, 3165, 896])  多图 torch.Size([1, 6578, 896])
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
