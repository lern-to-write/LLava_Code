from llava.model.token_select import *
from llava.model.compute_score_by_kernal import *
from llava.model.merge_background import *


def uniform_ratio_conbine_mean_key_half_big_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_big_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_big_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def uniform_ratio_conbine_mean_key_half_small_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def uniform_ratio_conbine_mean_key_rbf_kernal_v1(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_by_kernal(flattened_image_feature)
    video_mean_score=compute_video_mean_score_by_kernal(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices





def uniform_ratio_conbine_mean_key_rbf_kernal_v2(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_by_kernal_v2(flattened_image_feature)
    video_mean_score=compute_video_mean_score_by_kernal_v2(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def uniform_ratio_conbine_mean_key_mul_gaussion_kernal(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_multi_gaussian(flattened_image_feature)
    video_mean_score=compute_video_mean_score_multi_gaussian(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def uniform_ratio_conbine_mean_key_merge_background(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_double_specific_retention_big_merge_and_small_select(flattened_image_feature, scales, video_mean_score,frame_mean_score)
    return sorted_indices



def uniform_ratio_conbine_mean_key_merge_background_by_30_berfore(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_double_specific_retention_big_merge_and_small_select_30(flattened_image_feature, scales, video_mean_score,frame_mean_score)
    return sorted_indices



def uniform_ratio_frame_mean_key_merge_background(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention_big_merge_and_small_select(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices



def uniform_ratio_frame_mean_key_merge_background_by_30_berfore(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention_big_merge_and_small_select_30(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices

def uniform_ratio_single_mean_key_half_big_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_big_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_big_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


def uniform_ratio_single_mean_key_half_small_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


def uniform_ratio_single_mean_key_rbf_kernal_v1(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_by_kernal(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices