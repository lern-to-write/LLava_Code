import os
import subprocess
import warnings
from typing import List, Union, Tuple

import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from llava.conversation import conv_templates

# Suppress warnings
warnings.filterwarnings("ignore")


def load_model() -> Tuple:
    """
    Load and initialize the LLaVA model.
    
    Returns:
        tuple: (tokenizer, model, image_processor, max_length)
    """
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    model_name = "llava_qwen"
    device_map = "auto"
    
    llava_model_args = {
        "multimodal": True,
    }
    
    return load_pretrained_model(
        pretrained, 
        None, 
        model_name, 
        device_map=device_map,
        attn_implementation="sdpa",
        **llava_model_args
    )

def load_video(video_path: Union[str, List[str]], max_frames_num: int) -> np.ndarray:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file or list containing video path
        max_frames_num: Number of frames to extract
        
    Returns:
        np.ndarray: Extracted video frames
    """
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
        
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    return vr.get_batch(frame_idx).asnumpy()

def process_video(video_path: str, question: str = None) -> str:
    """
    Process video and generate description.
    
    Args:
        video_path: Path to the video file
        question: Custom question for the model (optional)
        
    Returns:
        str: Generated description of the video
    """

    
    # Load model
    tokenizer, model, image_processor, max_length = load_model()
    model.eval()
    
    # Process video
    video_frames = load_video(video_path, 16)


    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    
    image_tensors.append(frames)
    
    # Prepare conversation
    if question is None:
        #question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
        question = f"{DEFAULT_IMAGE_TOKEN}\nWho are all the people in the video?? "
    
    conv_template = "qwen_1_5"
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # Prepare inputs
    input_ids = tokenizer_image_token(
        prompt_question, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to("cuda")
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

if __name__ == "__main__":
    video_path = "/root/autodl-tmp/LLaVA-NeXT/docs/jobs.mp4"
    result = process_video(video_path)
    print(result)