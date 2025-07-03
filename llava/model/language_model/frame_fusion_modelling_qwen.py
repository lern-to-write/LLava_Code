from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from llava.model.data_store import data

import logging
import pdb
from llava.model.language_model.frame_wise_compression import *
# 配置日志记录器
logging.basicConfig(
    filename='/root/autodl-tmp/LLaVA-NeXT/llava/model/language_model/qwen2_model.log',  # 日志文件路径
    level=logging.DEBUG,         # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# logger = logging.get_logger(__name__)
logger = logging.getLogger(__name__)


class FastVQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        self.last_attention = None
        super().__init__(config)
    def forword(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        model = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        ### change position_embeddings into a list for future pruning
        position_embeddings = list(position_embeddings) 

        ### end changing

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        ### implement fastv
        FASTV_k = 3 # the layer_idx to prune
        FASTV_r = 5 # the pruning ratio
        FASTV_image_token_start_index = 15
        FASTV_image_token_length = 3136
        device = self.device

        frame_token_length=data.frame_token_length
        all_image_token_length=data.all_image_token_length
        frame_number=data.frame_number
        system_token_length=data.system_token_length
        
        #seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1] (here because cache position in minicpmv is not none,so past_seen_tokens is not defined )
        #for layer_idx, decoder_layer in enumerate(self.layers):
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # pruning hidden states, no kv cache
            
            if use_cache:
                print("use_cache")
                if hidden_states.shape[1] != 1:
                    print("hidden_states.shape[1] != 1")
                    if decoder_layer.self_attn.layer_idx <FASTV_k:
                        
                        pruned_attention_mask = causal_mask

                    elif decoder_layer.self_attn.layer_idx == 3 :
                        print("use frame wise compression")

                        # compute pruned tokens, generate fastv sign
                        last_layer_attention = layer_outputs[1]
                        # compute average attention over different head
                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                        # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        # get the attention in image token
                        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[FASTV_image_token_start_index:FASTV_image_token_start_index+FASTV_image_token_length]
                        # get the indexs of the top ATTENTION_RANK tokens
                        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(round(FASTV_image_token_length*(1-FASTV_r))).indices + FASTV_image_token_start_index
                        # keep index
                        keep_indexs = torch.cat( (torch.arange(FASTV_image_token_start_index,device=device), top_attention_rank_index, torch.arange(FASTV_image_token_start_index+FASTV_image_token_length,hidden_states.shape[1],device=device)))
                        # sort index
                        keep_indexs = keep_indexs.sort().values
                        # update seq length
                        new_seq_length = keep_indexs.shape[0]
                        # filter hidden states
                        hidden_states = hidden_states[:,keep_indexs,:] 
                        # update position ids
                        position_ids = keep_indexs.unsqueeze(0)
                        # update position embeddings
                        position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                        position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]

                        cache_position = cache_position[:new_seq_length]
                        logging.debug(f"cache_position{cache_position.shape}hidden_states: {hidden_states.shape}")
                        # retention_ratio=0.25
                        # last_layer_attention = layer_outputs[1]
                        # score_in_local_frame=get_score_in_local_frame(last_layer_attention, all_image_token_length, frame_token_length, frame_number, system_token_length)
                        # score_in_globle=get_score_in_globle(last_layer_attention, all_image_token_length, frame_number, frame_token_length, system_token_length)
                        # scales=generate_scale_for_frame(last_layer_attention, all_image_token_length, frame_number, frame_token_length, system_token_length,base_scale=retention_ratio, temperature=10.0)
                        # keep_indexs=select_topk_token_index_each_frame(hidden_states, score_in_globle, all_image_token_length, score_in_local_frame,frame_number,system_token_length, frame_token_length, scales)

                        # # update seq length
                        # new_seq_length = keep_indexs.shape[0]
                        # # filter hidden states
                        # hidden_states = hidden_states[:,keep_indexs,:] 
                        # # update position ids
                        # position_ids = keep_indexs.unsqueeze(0)
                        # # update position embeddings
                        # position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                        # position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]

                        # cache_position = cache_position[:new_seq_length]
                        

                else:
                    pruned_attention_mask = causal_mask
                    print("not use frame wise compression")

            else:
                raise NotImplementedError("fastv only support use_cache=True")
    

            if decoder_layer.self_attn.layer_idx == FASTV_k - 1:
                output_attentions = True
            else:
                output_attentions = False
            logging.debug(f"output_attentions{output_attentions.shape}hidden_states: {hidden_states.shape}")
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=pruned_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
            )
        ### end fastv

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        ### force output_attentions to be False(we store attn_weights by ourselves)
            output_attentions = False
        ### end
        
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
