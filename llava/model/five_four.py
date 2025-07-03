from llava.model.token_select import *
from llava.model.five_four_double_trick import *

from llava.model.compute_score_by_kernal import *
from llava.model.merge_background import *



def orign_score_and_scale_video_sum_conbine_token_positive_half_channel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def orign_score_and_scale_video_sum_conbine_token_negitive_half_cahnnel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices
def equal_score_and_scale_video_sum_conbine_token_negitive_half_cahnnel(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169

    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices
def orign_score_and_scale_two_mean_conbine_token_positive_positive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=compute_frame_score_by_two_mean(flattened_image_feature)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices
def orign_score_and_scale_two_mean_conbine_token_positive_negitive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=-compute_frame_score_by_two_mean(flattened_image_feature)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices




############################################################
def negitive_frame_score_conbine_token_multi_gassion_matrix_half_cahnnel_25(flattened_image_feature):
    
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.25)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def negitive_frame_score_conbine_token_multi_gassion_matrix_half_cahnnel_15(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.15)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def negitive_frame_score_conbine_token_multi_gassion_matrix_half_cahnnel_30(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def vidcom2_compression(flattened_image_feature,model,base_scale,frame_score_mode):
    print()
    if model == "llava_ov":
        token_per_frame = 196
    elif model == "llava_vid":
        token_per_frame = 169
    else :
        raise ValueError("Invalid model name. Choose from 'llava_ov', 'llava_ov_25'.")
    if frame_score_mode == "equal":
        num_frames = flattened_image_feature//token_per_frame
        scales = torch.full((num_frames,), base_scale)
    elif frame_score_mode == "negtive":
        selected_tensor=select_feature_channel(flattened_image_feature)
        frame_mean_score=compute_frame_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=token_per_frame)
        video_mean_score=compute_video_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=token_per_frame)
        frame_score=-video_mean_score.mean(-1)
        scales=generate_scales_from_frame_video_score(frame_score,base_scale=base_scale)
    elif frame_score_mode == "positive":
        selected_tensor=select_feature_channel(flattened_image_feature)
        frame_mean_score=compute_frame_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=token_per_frame)
        video_mean_score=compute_video_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=token_per_frame)
        frame_score=video_mean_score.mean(-1)
        scales=generate_scales_from_frame_video_score(frame_score,base_scale=base_scale)
    else:
        raise ValueError("Invalid frame_score_mode. Choose from 'equal', 'negtive', or 'positive'.")
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score,token_per_frame=token_per_frame)

    if model == "llava_ov":
        converted_indice=convert_to_global_indices_tensor(sorted_indices)
        converted_image_feature=flattened_image_feature[converted_indice]
    elif model == "llava_vid":

        keep_index=get_flattened_keep_indices(sorted_indices,resize_h=13)
        converted_image_feature=flattened_image_feature[keep_index]
    else:
        raise ValueError("Invalid model name. Choose from 'llava_ov', 'llava_vid'.")

    return converted_image_feature


def negitive_frame_score_conbine_token_multi_gassion_matrix_half_channel_llava_ov_25(flattened_image_feature):
    selected_tensor=select_feature_channel(flattened_image_feature)
    frame_mean_score=compute_frame_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=196)
    video_mean_score=compute_video_mean_score_multi_gaussian_matrix(selected_tensor, token_per_frame=196)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.25)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score,token_per_frame=196)
    return sorted_indices
def negitive_frame_score_conbine_token_multi_gassion_matrix_half_channel_llava_ov_15(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature,token_per_frame=196)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature,token_per_frame=196)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.15)  
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score,token_per_frame=196)
    return sorted_indices

def negitive_frame_score_conbine_token_multi_gassion_matrix_half_channel_llava_ov_30(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature,token_per_frame=196)
    video_mean_score=compute_video_mean_score_matrix_multi_gaussian_small_var_channel(flattened_image_feature,token_per_frame=196)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=0.3)  
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score,token_per_frame=196)
    return sorted_indices
def convert_to_global_indices_tensor(indices_list, token_per_frame=196):

    device = indices_list[0].device
    num_frames = len(indices_list)

    frame_indices = torch.arange(num_frames, device=device).unsqueeze(1)

    frame_offsets = frame_indices * token_per_frame
    global_indices = torch.cat([indices + offset for indices, offset in zip(indices_list, frame_offsets)])
    return global_indices