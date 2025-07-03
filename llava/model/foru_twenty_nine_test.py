from llava.model.token_select import *
from llava.model.data_store import data

# 原来的最work的  token score使用正常一般channel的 ；frame score使用正的video mean score；LLM内使用conbine score
# 使用核函数v1的作为frame score 正的
# 使用核函数v1的作为frame score 负的


# 使用自适应模态ratio（模态打分方案）+原来的最work的  token score使用正常一般channel的 ；frame score使用正的video mean score；LLM内使用conbine score
def orign_score_and_scale(flattened_image_feature):
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    data.vision_scale=base_scale
    return sorted_indices




def kernal_positive_score_and_scale(flattened_image_feature):
    base_scale=0.5
    frame_mean_score=compute_frame_mean_score_by_kernal(flattened_image_feature)
    video_mean_score=compute_video_mean_score_by_kernal(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    scales=generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    data.vision_scale=base_scale

    return sorted_indices

def kernal_negtive_score_and_scale(flattened_image_feature):
    base_scale=0.5
    frame_mean_score=compute_frame_mean_score_by_kernal(flattened_image_feature)
    video_mean_score=compute_video_mean_score_by_kernal(flattened_image_feature)
    frame_score=-video_mean_score.mean(-1)
    scales= generate_scales_from_frame_video_score(frame_score)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    data.vision_scale=base_scale

    return sorted_indices

def orign_score_and_scale_mode_score_big(flattened_image_feature):
    # 相似度越大保留率越大
    base_score=1
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    vision_mode_score=video_mean_score.mean()
    base_scale=vision_mode_score/base_score
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=base_scale)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    data.vision_scale=base_scale

    return sorted_indices


def orign_score_and_scale_mode_score_small(flattened_image_feature):
    # 相似度越小保留率越大
    base_score=1
    frame_mean_score=compute_frame_mean_score_small_var(flattened_image_feature)
    video_mean_score=compute_video_mean_score_small_var(flattened_image_feature)
    frame_score=video_mean_score.mean(-1)
    vision_mode_score=video_mean_score.mean()
    base_scale=vision_mode_score/base_score
    scales=generate_scales_from_frame_video_score(frame_score,base_scale=base_scale)
    sorted_indices=get_indeice_of_select_token_with_conbine_specific_retention(flattened_image_feature, scales,video_mean_score,frame_mean_score)
    data.vision_scale=base_scale

    return sorted_indices

