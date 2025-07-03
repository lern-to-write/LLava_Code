from llava.model.token_select import *
from llava.model.compute_score_by_kernal import *
from llava.model.merge_background import *



def orign_score_and_scale_video_sum_conbine_key_positive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def orign_score_and_scale_two_mean_conbine_key_positive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=compute_frame_score_by_two_mean(flattened_image_feature)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices



def orign_score_and_scale_video_sum_conbine_key_negtive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices


def orign_score_and_scale_two_mean_conbine_key_negtive(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=-compute_frame_score_by_two_mean(flattened_image_feature)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices