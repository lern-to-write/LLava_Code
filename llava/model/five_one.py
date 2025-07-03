from llava.model.token_select import *

def uniform_ratio_with_average_frame_token(flattened_image_feature):
    # 帧间uniform ratio +  帧内token with average frame token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices


def uniform_ratio_with_average_frame_big_and_small_token(flattened_image_feature):
    # 帧间uniform ratio +  帧内token with average frame token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_single_specific_retention_big_and_small(flattened_image_feature, scales,frame_mean_score)
    return sorted_indices



def uniform_ratio_with_average_conbine_big_and_small_token(flattened_image_feature):
    # 帧间uniform ratio +  帧内token with average frame token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)

    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_double_specific_retention_big_and_small(flattened_image_feature, scales,frame_mean_score,video_mean_score)
    return sorted_indices

def uniform_ratio_with_average_video_add_frame_token(flattened_image_feature):
    # 帧间uniform ratio +  帧内token with average frame token + 帧内token with average video token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)

    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def uniform_ratio_with_average_video_subtract_frame_token(flattened_image_feature):
    # 帧间uniform ratio -  帧内token with average frame token + 帧内token with average video token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)

    num_frames=flattened_image_feature.shape[0]//169
    scales = torch.full((num_frames,), 0.3)
    sorted_indices=get_indeice_of_select_token_with_subtract_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

def conbine_score_ratio_frame_mean_and_video_mean(flattened_image_feature):
    # 帧间dynamic ratio（每帧average成一个token与整个视频average成一个token的相似度） + 帧内token with average frame token + 帧内token with average video token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=compute_frame_score_by_two_mean(flattened_image_feature)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices



def conbine_score_ratio_video_mean(flattened_image_feature):


    # 帧间dynamic ratio（每帧average成一个token与整个视频average成一个token的相似度） + 帧内token with average frame token + 帧内token with average video token
    frame_mean_score=compute_frame_mean_score(flattened_image_feature)
    video_mean_score=compute_video_mean_score(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    return sorted_indices

