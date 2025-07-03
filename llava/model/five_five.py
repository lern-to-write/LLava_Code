from llava.model.token_select import *
from llava.model.compute_score_by_kernal import *
from llava.model.merge_background import *
from llava.model.five_four_double_trick import *



def uniform_ratio_frame_mean(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices

def uniform_ratio_frame_mean_small_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


def negtive_adaptive_ratio_frame_mean_half_small_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices

def positive_adaptive_ratio_frame_mean_half_small_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices

def positive_adaptive_ratio_frame_mean(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices



def negtive_adaptive_ratio_frame_mean(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices



def positive_adaptive_ratio_frame_mean_multi_gassion_matrix_half_cahnnel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


def negtive_adaptive_ratio_frame_mean_multi_gassion_matrix_half_cahnnel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


