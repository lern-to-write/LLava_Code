# main.py
import warnings
import requests
import copy
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle

def main():
    # 忽略警告信息
    warnings.filterwarnings("ignore")

    # 模型配置
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }

    # 加载模型
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        device_map=device_map,
        **llava_model_args
    )

    # 设置模型为评估模式
    model.eval()

    # 加载和处理图像
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    image_tensor = process_images([image], image_processor, model.config)
    
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    print("image_tensor", image_tensor.shape)

    # 准备对话模板
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 准备输入
    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_sizes = [image.size]

    # 生成回答
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )

    # 解码并打印输出
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)

if __name__ == "__main__":
    main()